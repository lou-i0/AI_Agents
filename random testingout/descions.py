

# %%
from openai import OpenAI
# %% 
MODEL = "gpt-6-luna"

# %%
def main() -> None:
    with OpenAI(timeout=20.0) as client:
        result = client.decisions.create(
            model=MODEL,
            input="My keyboard arrived with three broken keys. Can you replace it?",
            questions=[{
                "type": "choice",
                "name": "queue",
                "instructions": "Choose the queue for the customer's main request.",
                "choices": [
                    {"value": "returns", "description": "Damaged goods or replacements."},
                    {"value": "billing", "description": "Charges or invoice errors."},
                    {"value": "delivery", "description": "Missing or delayed deliveries."},
                    {"value": "general", "description": "Everything else or unclear intent."},
                ],
            }],
        )

        answer = result.answers[0]
        if answer.type == "refusal":
            print("Send to manual triage")
        elif answer.type == "choice":
            print(answer.choice, answer.confidence)
            queue = answer.choice if answer.confidence >= 0.8 else "manual_triage"
            print("Queue:", queue)


if __name__ == "__main__":
    main()



