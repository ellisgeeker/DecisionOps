#!/usr/bin/env python3
"""
Build a larger synthetic decision dataset for the technical gate.

Uses a sklearn text classifier so probabilities come from a real model,
then deliberately sharpens them (T<1) to create miscalibration.

Fully offline: no network downloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

INTENTS = [
    "billing_refund",
    "billing_invoice",
    "tech_login",
    "tech_outage",
    "account_cancel",
    "account_upgrade",
    "shipping_track",
    "shipping_return",
]

# Intentionally overlapping language so the task is non-trivial.
TEMPLATES = {
    "billing_refund": [
        "charged twice for order {n}, want my money back",
        "refund the payment on order {n} please",
        "billing error double charge order {n}",
        "cancel accidental purchase and refund order {n}",
        "card charged wrongly, refund order {n}",
    ],
    "billing_invoice": [
        "need invoice for order {n}",
        "send VAT receipt payment {n}",
        "billing statement for order {n}",
        "download invoice PDF order {n}",
        "receipt copy for charge {n}",
    ],
    "tech_login": [
        "cannot log into account {n}",
        "password reset failed for user {n}",
        "locked out of login account {n}",
        "2fa code not working login {n}",
        "sign-in page error account {n}",
    ],
    "tech_outage": [
        "service down right now status {n}",
        "API 503 errors since morning {n}",
        "dashboard not loading possible outage {n}",
        "requests timing out region {n}",
        "system unavailable error {n}",
    ],
    "account_cancel": [
        "cancel my subscription plan {n}",
        "close account permanently {n}",
        "stop auto-renew and cancel {n}",
        "unsubscribe membership {n}",
        "delete account and cancel billing {n}",
    ],
    "account_upgrade": [
        "upgrade to pro plan {n}",
        "need more seats team tier {n}",
        "switch free to business {n}",
        "increase quota enterprise {n}",
        "move account to higher plan {n}",
    ],
    "shipping_track": [
        "where is package shipment {n}",
        "tracking no updates order {n}",
        "parcel left warehouse order {n}",
        "delivery status order {n}",
        "track my shipment {n}",
    ],
    "shipping_return": [
        "return this item order {n}",
        "start RMA damaged goods {n}",
        "send return label package {n}",
        "wrong size process return {n}",
        "return shipping for order {n}",
    ],
}

SHARED_NOISE = [
    "please help",
    "urgent",
    "customer account",
    "order issue",
    "thanks",
    "app problem",
    "billing and shipping related",
    "not sure where to route this",
]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def build_corpus(n_per_class: int, rng: np.random.Generator) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for intent in INTENTS:
        for i in range(n_per_class):
            tmpl = TEMPLATES[intent][i % len(TEMPLATES[intent])]
            noise_id = int(rng.integers(1000, 9999))
            text = tmpl.format(n=noise_id)
            # Mix shared phrases to reduce separability.
            if rng.random() < 0.55:
                text = text + " " + str(rng.choice(SHARED_NOISE))
            if rng.random() < 0.35:
                text = str(rng.choice(SHARED_NOISE)) + " " + text
            # Occasional cross-intent phrase injection.
            if rng.random() < 0.2:
                other = INTENTS[int(rng.integers(0, len(INTENTS)))]
                text = text + " " + TEMPLATES[other][0].format(n=int(rng.integers(10, 99)))
            texts.append(text)
            labels.append(intent)
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.35)
    parser.add_argument("--label-noise", type=float, default=0.08)
    parser.add_argument(
        "--sharpen",
        type=float,
        default=0.5,
        help="T<1 sharpens probabilities (overconfidence).",
    )
    parser.add_argument("--out", type=Path, default=Path("examples/gate-synthetic"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    texts, labels = build_corpus(args.n_per_class, rng)
    n = len(texts)

    # Inject label noise into training targets only later; also noise some gold labels lightly
    # for realism on a fraction of rows (kept in gold for eval hardness via model mismatch).
    idx = rng.permutation(n)
    train_n = max(50, int(args.train_frac * n))
    train_idx = idx[:train_n]

    y_train = []
    for i in train_idx:
        y = labels[i]
        if rng.random() < args.label_noise:
            y = INTENTS[int(rng.integers(0, len(INTENTS)))]
        y_train.append(y)

    clf = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
            ("lr", LogisticRegression(max_iter=2000, C=0.8, random_state=args.seed)),
        ]
    )
    clf.fit([texts[i] for i in train_idx], y_train)

    proba = clf.predict_proba(texts)
    classes = list(clf.named_steps["lr"].classes_)

    logits = np.log(np.clip(proba, 1e-12, 1.0))
    sharp = _softmax(logits / args.sharpen)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    ds_path = out / "dataset.jsonl"
    pr_path = out / "predictions.jsonl"

    with ds_path.open("w", encoding="utf-8") as fds, pr_path.open("w", encoding="utf-8") as fpr:
        for i in range(n):
            cid = f"intent-{i:04d}"
            label = labels[i]
            probs = {c: float(sharp[i, j]) for j, c in enumerate(classes)}
            s = sum(probs.values())
            probs = {k: v / s for k, v in probs.items()}
            pred = max(probs, key=probs.get)

            fds.write(
                json.dumps(
                    {
                        "id": cid,
                        "state": texts[i],
                        "question": "Which support intent should we route to?",
                        "choices": INTENTS,
                        "label": label,
                        "metadata": {"domain": "support-intent", "synthetic": True},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fpr.write(
                json.dumps(
                    {
                        "id": cid,
                        "prediction": pred,
                        "probabilities": probs,
                        "latency_ms": round(float(rng.uniform(8.0, 35.0)), 1),
                        "provider": "static",
                        "model": f"sklearn-lr-sharpen{args.sharpen}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    preds = sharp.argmax(axis=1)
    class_index = {c: j for j, c in enumerate(classes)}
    acc = np.mean([preds[i] == class_index[labels[i]] for i in range(n)])
    print(f"wrote n={n} to {out}")
    print(f"accuracy(raw sharpened)={acc:.4f}")
    print(f"mean_max_prob={sharp.max(axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
