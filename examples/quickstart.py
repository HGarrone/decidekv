"""kvindex quickstart. Before running: `ollama pull qwen3:8b` (any pure-attention GGUF model works).

    python examples/quickstart.py
"""
import time

from kvindex import KV, Choice, Noul, Score


def main():
    kv = KV("qwen3:8b")
    try:
        # 1) An index over a fixed option set. Building it fills the model's KV cache once.
        intents = kv.index({
            "card_arrival": "the customer is waiting for a card to be delivered",
            "lost_or_stolen_card": "the card was lost or stolen",
            "refund_not_showing_up": "a refund has not appeared on the account",
            "exchange_rate": "questions about currency exchange rates",
            "transfer_not_received": "a transfer did not reach the recipient",
        }, task="Classify bank customer messages.")

        # 2) Each decision computes only the new message: the options are already in the cache.
        for msg in ["I ordered my card two weeks ago and it's still not here",
                    "Someone took my wallet with the card inside",
                    "I sent money to my brother yesterday but he still hasn't received it"]:
            t = time.perf_counter()
            a = intents.choose(msg)
            print(f"{a.choice:<24} conf={a.confidence:.2f}  {1000 * (time.perf_counter() - t):.0f} ms  <- {msg}")

        # 3) OpenJev / Jev-style: several typed questions about one state, in one call to decide().
        r = kv.decide(
            state={"customer": "This is the third time I write! My card was stolen yesterday and nobody answers."},
            questions={
                "intent": Choice(instructions="What is the customer's request?", options=intents),
                "mood": Score(instructions="How upset is the customer?", criteria=["calm", "annoyed", "furious"]),
                "escalate": Noul(instructions="Should this be escalated to a human right now?"),
            },
        )
        print(r.answers["intent"].choice, r.answers["mood"].score, r.answers["escalate"].noul, f"{r.latency_ms} ms")
        print(r.usage)
    finally:
        kv.close()


if __name__ == "__main__":
    main()
