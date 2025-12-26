import os
import time
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict, Tuple

load_dotenv()


# Askri Bank Limited Knowledge Base
ASKRI_BANK_KNOWLEDGE_BASE = """
# ASKRI BANK LIMITED - COMPLETE KNOWLEDGE BASE

## 1. BANK OVERVIEW
- **Bank Name:** Askri Bank Limited
- **Customer Service Hotline:** 1-800-ASKRI (1-800-27574)
- **Operating Hours:** 24/7 for phone banking, Branches: Mon-Fri 9AM-5PM, Sat 9AM-1PM
- **Website:** www.asskribank.com
- **Mobile App:** Askri Bank Mobile (iOS & Android)

## 2. ACCOUNT TYPES

### 2.1 Savings Accounts
| Account Type | Min Balance | APY | Monthly Fee | Features |
|-------------|-------------|-----|-------------|----------|
| Basic Savings | $100 | 0.50% | $5 (waived if >$500) | 3 free ATM/month |
| Premium Savings | $1,000 | 2.50% | $0 | Unlimited ATM, free checks |
| High-Yield Savings | $10,000 | 4.25% | $0 | Priority service |

### 2.2 Checking Accounts
| Account Type | Min Balance | Monthly Fee | Features |
|-------------|-------------|-------------|----------|
| Basic Checking | $0 | $8 (waived with DD or $1,500) | Free debit card, bill pay |
| Premium Checking | $5,000 | $0 | 0.25% APY, free checks |
| Student Checking | $0 (ages 17-24) | $0 | Overdraft forgiveness |

## 3. CREDIT CARDS

### Askri Rewards Card
- Annual Fee: $0 | APR: 18.99%-24.99%
- Rewards: 1.5% cash back on all purchases
- Sign-up Bonus: $200 after $500 spend in 3 months

### Askri Travel Elite Card
- Annual Fee: $95 | APR: 17.99%-23.99%
- Rewards: 3x travel, 2x dining, 1x other
- Perks: Lounge access, travel insurance

### Askri Secured Card
- For building credit | Deposit: $200-$2,500 | APR: 22.99%

## 4. LOANS

### Personal Loans
- Amount: $1,000-$50,000 | APR: 7.99%-19.99% | Terms: 12-60 months

### Auto Loans
- New Car: From 5.49% | Used Car: From 6.49%
- Terms: 24-84 months | Up to 100% financing

### Mortgages
- 30-Year Fixed: From 6.75% | 15-Year Fixed: From 6.25%
- FHA, VA, and refinancing available

## 5. FEES SCHEDULE

| Fee Type | Amount |
|----------|--------|
| Overdraft | $35/occurrence |
| NSF | $35 |
| Stop Payment | $30 |
| Domestic Wire (Out) | $25 |
| International Wire (Out) | $45 |

## 6. DIGITAL BANKING

### Mobile App Features
- Mobile deposit: $5,000/day
- Biometric login, card lock/unlock
- Apple Pay, Google Pay, Samsung Pay

## 7. COMMON ISSUES

### Lost/Stolen Card
1. Lock card via app
2. Call customer service
3. New card in 5-7 days

### Password Reset
1. Click "Forgot Password"
2. Verify via security questions
3. Create new password

## 8. CONTACT
- **Phone:** 1-800-ASKRI (24/7)
- **Chat:** Online banking or mobile app
"""


class BankingLLM:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """Initialize the Banking LLM."""
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 20

        self.last_latency_ms = 0
        self.last_tokens = 0

        # System prompt optimized for voice and low latency
        self.system_prompt = f"""You are a customer support agent for Askri Bank Limited.
Keep responses brief (1-2 sentences). Be helpful.
Never ask for SSN, passwords, or PINs.

{ASKRI_BANK_KNOWLEDGE_BASE}

Be concise."""

    def get_response(self, user_message: str) -> Tuple[str, dict]:
        """Get a response from the LLM."""
        start_time = time.perf_counter()

        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history[-self.max_history:]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=100,
                top_p=0.85,
                stream=False
            )

            latency = (time.perf_counter() - start_time) * 1000

            assistant_message = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            self.last_latency_ms = latency
            self.last_tokens = tokens_used

            return assistant_message, {"latency_ms": latency, "tokens": tokens_used}

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            print(f"LLM Error: {e}")
            return "Sorry, please try again.", {"latency_ms": latency, "error": str(e)}

    def get_response_streaming(self, user_message: str):
        """Get streaming response for lower perceived latency."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history[-self.max_history:]

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=100,
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text

            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:
            print(f"LLM Error: {e}")
            yield "Sorry, please try again."

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []

    def get_greeting(self) -> Tuple[str, dict]:
        """Get initial greeting."""
        start_time = time.perf_counter()

        greeting = "Hello! Welcome to Askri Bank Limited. How can I help you today?"

        self.conversation_history.append({
            "role": "assistant",
            "content": greeting
        })

        return greeting, {"latency_ms": (time.perf_counter() - start_time) * 1000}


if __name__ == "__main__":
    print("Testing LLM...")

    llm = BankingLLM(model="llama-3.1-8b-instant")

    greeting, metrics = llm.get_greeting()
    print(f"Greeting: {greeting}")

    test_queries = [
        "What's your best savings rate?",
        "I lost my card"
    ]

    for query in test_queries:
        print(f"\nQ: {query}")
        response, metrics = llm.get_response(query)
        print(f"A: {response}")
        print(f"[{metrics['latency_ms']:.0f}ms]")