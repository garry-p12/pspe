"""Phase 4: brief rendering, the frozen parser, and the faithfulness objective."""

from __future__ import annotations

import json

import torch

from pspe.envs import make_env
from pspe.explain import (
    BriefContext,
    ExplainConfig,
    ExplainTrainConfig,
    ExplainTrainer,
    FrozenBriefParser,
    WordTokenizer,
    build_vocabulary,
    faithfulness_score,
    render_brief,
)
from pspe.explain.faithfulness import FaithfulnessObjective
from pspe.plan import GaussianFieldPolicy
from pspe.utils import RunLogger

GRID = 16


def _context(amps: torch.Tensor, centers: torch.Tensor) -> BriefContext:
    return BriefContext(
        step=3, amplitudes=amps, centers=centers, predicted_cost=1.2,
        cost_limit=2.0, predicted_reward=-0.4,
        log_std=torch.full((amps.numel(),), -2.0),
    )


def test_reference_brief_round_trips_through_the_parser() -> None:
    """A rendered brief must parse back to the amplitudes it names."""
    centers = torch.rand(4, 2)
    amps = torch.tensor([0.40, -0.25, 0.05, 0.0])
    brief = render_brief(_context(amps, centers))

    parser = FrozenBriefParser(4)
    mean, std, extras = parser.parse(brief)
    assert mean[0] == 0.40 and mean[1] == -0.25
    # 0.05 is at the deadband, 0.0 is below it: neither is named, both parse to 0.
    assert mean[3] == 0.0
    assert extras["mentioned_actuators"] >= 2
    assert float(std[0]) > 0.0


def test_brief_reports_the_constraint_status() -> None:
    centers = torch.rand(3, 2)
    safe = render_brief(
        BriefContext(1, torch.tensor([0.3, 0.0, 0.0]), centers, 0.5, 2.0, -0.1)
    )
    unsafe = render_brief(
        BriefContext(1, torch.tensor([0.3, 0.0, 0.0]), centers, 9.0, 2.0, -0.1)
    )
    assert "within the limit" in safe
    assert "above the limit" in unsafe


def test_empty_plan_produces_a_hold_brief() -> None:
    brief = render_brief(BriefContext(0, torch.zeros(4), torch.rand(4, 2), 0.1, 2.0, 0.0))
    assert "hold all settings" in brief
    assert FrozenBriefParser(4).parse(brief)[0].abs().sum() == 0.0


def test_faithfulness_is_one_for_an_exact_brief_and_falls_off() -> None:
    centers = torch.rand(4, 2)
    amps = torch.tensor([0.40, -0.25, 0.0, 0.0])
    brief = render_brief(_context(amps, centers))
    parser = FrozenBriefParser(4)
    parsed, _ = parser.batch_distribution([brief])

    exact = torch.distributions.Normal(amps.unsqueeze(0), parsed.stddev)
    wrong = torch.distributions.Normal(-amps.unsqueeze(0), parsed.stddev)

    good, good_kl = faithfulness_score(exact, parsed)
    bad, bad_kl = faithfulness_score(wrong, parsed)
    assert float(good) == 1.0 and float(good_kl) == 0.0
    assert float(bad) < float(good)


def test_tokenizer_covers_the_brief_vocabulary() -> None:
    centers = torch.rand(9, 2)
    tokenizer = WordTokenizer(build_vocabulary(9, centers))
    brief = render_brief(_context(torch.tensor([0.35] + [0.0] * 8), centers))
    assert tokenizer.unknown_rate(brief) == 0.0
    ids, mask = tokenizer.batch_encode([brief])
    assert tokenizer.decode(ids[0]) == brief
    assert bool(mask[0].any())


def test_faithfulness_objective_drops_the_reinforce_term_when_ablated() -> None:
    objective = FaithfulnessObjective()
    nll = torch.tensor([1.0, 2.0])
    logprob = torch.tensor([-1.0, -3.0], requires_grad=True)
    score = torch.tensor([0.9, 0.1])

    _, off = objective(nll, logprob, score, use_faithfulness=False)
    _, on = objective(nll, logprob, score, use_faithfulness=True)
    assert off["loss/reinforce"] == 0.0
    assert off["loss/total"] == nll.mean().item()
    assert "metric/faithfulness" in on


def test_explain_trainer_logs_faithfulness_during_training(tmp_path) -> None:
    env = make_env("dar", grid=GRID, horizon=3, batched=True)
    policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)
    trainer = ExplainTrainer(
        env, policy,
        ExplainConfig(max_len=48, width=64, depth=2),
        ExplainTrainConfig(iterations=2, batch=2, log_dir=str(tmp_path / "explain")),
        RunLogger(tmp_path / "explain"),
    )
    summary = trainer.train()

    assert "eval/faithfulness" in summary
    assert 0.0 <= summary["eval/faithfulness"] <= 1.0

    # The acceptance check: F(b) is in the *training* log, not only the summary.
    lines = [
        json.loads(line)
        for line in (tmp_path / "explain" / "metrics.jsonl").read_text().splitlines()
    ]
    assert any("metric/faithfulness" in line for line in lines)

    briefs = (tmp_path / "explain" / "briefs.jsonl").read_text().splitlines()
    assert briefs and "generated" in json.loads(briefs[0])


def test_explain_stub_backbone_is_watermarked(tmp_path) -> None:
    env = make_env("dar", grid=GRID, horizon=2, batched=True)
    policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)
    trainer = ExplainTrainer(
        env, policy,
        ExplainConfig(max_len=32, width=64, depth=2),
        ExplainTrainConfig(iterations=1, batch=2, log_dir=str(tmp_path / "explain")),
        RunLogger(tmp_path / "explain"),
    )
    summary = trainer.train()
    assert summary["backbone_is_stub"] is True
    assert summary["backbone"] == "tiny"


def test_per_dim_kl_keeps_f_on_one_scale():
    """Summed KL sends F to 0 as the action space grows; per-dimension does not."""
    import torch
    from pspe.explain.faithfulness import faithfulness_score
    for k in (9, 64):
        p = torch.distributions.Normal(torch.zeros(1, k), torch.full((1, k), 0.22))
        q = torch.distributions.Normal(torch.full((1, k), 0.1), torch.full((1, k), 0.22))
        f_sum, kl_sum = faithfulness_score(p, q)
        f_dim, kl_dim = faithfulness_score(p, q, per_dim=True)
        assert abs(float(kl_dim) - float(kl_sum) / k) < 1e-6
        if k == 64:
            assert float(f_sum) < 0.01 and float(f_dim) > 0.5


def test_contrastive_loss_punishes_constant_briefs():
    """A brief that ignores the state pays; a brief that matches it does not."""
    import torch
    from pspe.explain.faithfulness import contrastive_faithfulness
    torch.manual_seed(0)
    b, k = 8, 9
    policy = torch.distributions.Normal(torch.randn(b, k), torch.full((b, k), 0.4))
    constant = torch.distributions.Normal(torch.zeros(b, k), torch.full((b, k), 0.4))
    perfect = torch.distributions.Normal(policy.mean.clone(), torch.full((b, k), 0.4))
    assert float(contrastive_faithfulness(policy, perfect).mean()) < 1e-4
    assert float(contrastive_faithfulness(policy, constant).mean()) > 2.0


def test_condition_dropout_only_in_training():
    import torch
    from pspe.explain.model import ExplainConfig, ExplainModule
    from pspe.explain.tokenizer import WordTokenizer
    tok = WordTokenizer(["<pad>", "<bos>", "<eos>", "<unk>", "a"])
    m = ExplainModule(tok, ExplainConfig(backbone="tiny", cond_dropout=1.0), cond_features=6)
    cond = torch.randn(4, 6)
    m.eval()
    assert not torch.allclose(m._prefix_embeds(cond), m._prefix_embeds(torch.zeros_like(cond)))
    m.train()
    assert torch.allclose(m._prefix_embeds(cond), m._prefix_embeds(torch.zeros_like(cond)))


def test_condition_contrastive_has_gradient_and_beats_log_b():
    """The differentiable contrastive term must reach the prefix parameters."""
    import math
    import torch
    from pspe.explain import ExplainConfig, ExplainTrainConfig
    from pspe.explain.model import ExplainModule
    from pspe.explain.tokenizer import WordTokenizer
    from pspe.explain.trainer import ExplainTrainer

    tok = WordTokenizer(["<pad>", "<bos>", "<eos>", "<unk>", "a", "b"])
    trainer = ExplainTrainer.__new__(ExplainTrainer)          # no env needed for this path
    trainer.cfg = ExplainTrainConfig(contrastive_temperature=1.0)
    trainer.model = ExplainModule(tok, ExplainConfig(backbone="tiny", cond_dropout=0.0),
                                  cond_features=6)
    torch.manual_seed(0)
    cond = torch.randn(4, 6)
    toks, mask = tok.batch_encode(["a b", "b a", "a a", "b b"], max_len=8)
    loss = trainer._condition_contrastive(cond, toks, mask.float())
    assert loss.shape == (4,)
    loss.mean().backward()
    prefix_grad = sum(float(p.grad.abs().sum()) for p in trainer.model.prefix.parameters()
                      if p.grad is not None)
    assert prefix_grad > 0, "contrastive loss does not reach the prefix"
    # An untrained model ignores the condition, so it should sit near log B.
    assert abs(float(loss.mean()) - math.log(4)) < 1.0
