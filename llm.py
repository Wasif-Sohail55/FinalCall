import os
import time
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict, Tuple

load_dotenv()


# XYZ Bank Knowledge Base
XYZ_BANK_KNOWLEDGE_BASE = """
# XYZ BANK - COMPLETE KNOWLEDGE BASE

## 1. BANK OVERVIEW
- **Bank Name:** XYZ Bank
- **Founded:** 1995
- **Headquarters:** New York City, USA
- **Customer Service Hotline:** 1-800-XYZ-BANK (1-800-999-2265)
- **Operating Hours:** 24/7 for phone banking, Branches: Mon-Fri 9AM-5PM, Sat 9AM-1PM
- **Website:** www.xyzbank.com
- **Mobile App:** XYZ Bank Mobile (iOS & Android)
- **Routing Number:** 021000089

## 2. ACCOUNT TYPES

### 2.1 Savings Accounts
| Account Type | Min Balance | APY | Monthly Fee | Features |
|-------------|-------------|-----|-------------|----------|
| XYZ Basic Savings | $100 | 0.50% | $5 (waived if >$500) | 3 free ATM/month |
| XYZ Premium Savings | $1,000 | 2.50% | $0 | Unlimited ATM, free checks |
| XYZ High-Yield Savings | $10,000 | 4.25% | $0 | Priority service, dedicated manager |

### 2.2 Checking Accounts
| Account Type | Min Balance | Monthly Fee | Features |
|-------------|-------------|-------------|----------|
| XYZ Basic Checking | $0 | $8 (waived with DD or $1,500) | Free debit card, bill pay |
| XYZ Premium Checking | $5,000 | $0 | 0.25% APY, free checks, no FX fees |
| XYZ Student Checking | $0 (ages 17-24) | $0 | Overdraft forgiveness (3x) |

## 3. CREDIT CARDS

### XYZ Rewards Card
- Annual Fee: $0 | APR: 18.99%-24.99%
- Rewards:  1.5% cash back on all purchases
- Sign-up Bonus: $200 after $500 spend in 3 months

### XYZ Travel Elite Card
- Annual Fee: $95 | APR:  17.99%-23.99%
- Rewards: 3x travel, 2x dining, 1x other
- Sign-up Bonus:  50,000 points after $3,000 in 3 months
- Perks:  Lounge access, travel insurance, no FX fees

### XYZ Secured Card
- For building credit | Deposit: $200-$2,500 | APR:  22.99%
- Upgrade to unsecured after 12 months good standing

## 4. LOANS

### Personal Loans
- Amount: $1,000-$50,000 | APR: 7.99%-19.99% | Terms: 12-60 months
- No origination fee, no prepayment penalty, funds in 1-2 days

### Auto Loans
- New Car:  From 5. 49% | Used Car: From 6.49%
- Terms: 24-84 months | Up to 100% financing
- 0.25% rate discount with auto-pay

### Mortgages
- 30-Year Fixed: From 6.75% | 15-Year Fixed: From 6.25%
- 5/1 ARM: From 5.99% | Down payment as low as 3%
- FHA, VA, and refinancing available

### HELOC
- APR: Prime + 0.50% (currently 8.50%)
- Draw:  10 years | Repayment: 20 years

## 5. FEES SCHEDULE

| Fee Type | Amount | Notes |
|----------|--------|-------|
| Overdraft | $35/occurrence | Max 3/day, first waived yearly |
| Overdraft Protection Transfer | $12 | |
| NSF | $35 | |
| Stop Payment | $30 | |
| Domestic Wire (Out) | $25 | |
| International Wire (Out) | $45 | |
| Wire (Incoming) | $15 | |
| Non-XYZ ATM | $3 + third-party | |
| Rush Card Replacement | $25 | Standard is free |
| Early Account Closure | $25 | Within 90 days |

**Fee Waivers:** Seniors (65+), military, 10+ year customers

## 6. DIGITAL BANKING

### Online Banking
- 24/7 access, bill pay, Zelle®, external transfers
- eStatements, budgeting tools, custom alerts

### Mobile App Features
- Mobile deposit:  $5,000/day, $25,000/month limits
- Biometric login, card lock/unlock, cardless ATM
- Apple Pay, Google Pay, Samsung Pay

### Security
- 2FA, real-time fraud monitoring, zero liability
- Secure messaging, instant account alerts

## 7. COMMON ISSUES & SOLUTIONS

### Lost/Stolen Card
1. Lock card immediately via app/online
2. Call 1-800-XYZ-BANK to report
3. Review transactions for fraud
4. New card in 5-7 days (rush available $25)

### Dispute Transaction
1. Log into online banking → Transaction history
2. Click transaction → "Dispute"
3. Provisional credit in 10 business days
4. Resolution in 45-90 days

### Password Reset
1. Click "Forgot Password" on login
2. Verify via security questions or SMS
3. Create new password
4. Call support if locked out

### Update Information
- Address/phone/email:  Online or branch
- Name change: Branch with legal docs
- SSN correction: Branch with documents

## 8. SPECIAL PROGRAMS

### XYZ Rewards
- 1 point per $2 on debit purchases
- Redeem:  Travel, merchandise, gift cards, statement credit
- Value: 1 point = $0.01 | Expire after 3 years inactivity

### Refer-a-Friend
- Earn $50 per referral (checking account)
- Friend gets $50 after first direct deposit
- Max $500/year

## 9. BUSINESS BANKING

### Business Accounts
- Basic: $0/month with $1,500 balance
- Plus: $25/month, 500 free transactions
- Premium: $50/month, unlimited transactions, free payroll

### Business Products
- Business Rewards Card: 2% cash back
- Term Loans: $10K-$500K
- Lines of Credit: $5K-$250K
- SBA Loans available

## 10. CONTACT INFORMATION

- **Phone:** 1-800-XYZ-BANK (1-800-999-2265) - 24/7
- **Chat:** Online banking or mobile app - 24/7
- **Email:** support@xyzbank. com (24hr response)
- **Social:** @XYZBank on Twitter, Facebook, Instagram
- **Branches:** Mon-Fri 9AM-5PM, Sat 9AM-1PM
"""


class BankingLLM:
    def __init__(self, model:  str = "llama-3.1-8b-instant"):
        """
        Initialize the Banking LLM.

        Args:
            model:  Groq model to use.  Options:
                - "llama-3.1-8b-instant" (fastest, good quality)
                - "llama-3.1-70b-versatile" (best quality, slower)
                - "mixtral-8x7b-32768" (balanced)
                - "gemma2-9b-it" (good for conversation)
        """
        self.client = Groq(api_key=os. getenv("GROQ_API_KEY"))
        self.model = model
        self.conversation_history:  List[Dict[str, str]] = []
        self. max_history = 20

        # Latency tracking
        self.last_latency_ms = 0
        self.last_tokens = 0

        # System prompt
        self. system_prompt = f"""You are Alex, a friendly and professional AI customer support agent for XYZ Bank. 
You are having a VOICE conversation, so keep responses conversational and concise. 

GUIDELINES:
1. Be warm, empathetic, and professional
2. Keep responses brief (2-4 sentences for simple queries)
3. For complex topics, break into digestible pieces
4. Verify concerns before providing solutions
5. If unsure, offer to connect with human agent
6. NEVER ask for full SSN, passwords, or PINs
7. Use natural speech, avoid reading URLs unless asked
8. Guide users to secure methods for sensitive actions

KNOWLEDGE BASE:
{XYZ_BANK_KNOWLEDGE_BASE}

Remember: You're SPEAKING, not writing. Be natural! """

    def get_response(self, user_message: str) -> Tuple[str, dict]:
        """
        Get a response from the LLM.

        Args:
            user_message:  User's transcribed speech

        Returns:
            Tuple of (response_text, metrics)
        """
        start_time = time. perf_counter()

        # Add to history
        self.conversation_history.append({
            "role":  "user",
            "content": user_message
        })

        # Prepare messages
        messages = [
            {"role": "system", "content": self. system_prompt}
        ] + self.conversation_history[-self.max_history:]

        try:
            response = self.client. chat.completions. create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=250,
                top_p=0.9,
                stream=False
            )

            latency = (time. perf_counter() - start_time) * 1000

            assistant_message = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            # Add to history
            self.conversation_history. append({
                "role": "assistant",
                "content":  assistant_message
            })

            self.last_latency_ms = latency
            self.last_tokens = tokens_used

            metrics = {
                "latency_ms":  latency,
                "tokens":  tokens_used,
                "model": self.model
            }

            return assistant_message, metrics

        except Exception as e:
            latency = (time. perf_counter() - start_time) * 1000
            print(f"LLM Error: {e}")
            return "I apologize, I'm having technical difficulties. Please try again.", {
                "latency_ms": latency,
                "error": str(e)
            }

    def get_response_streaming(self, user_message: str):
        """
        Get streaming response for lower perceived latency.

        Yields:
            Chunks of response text
        """
        self.conversation_history. append({
            "role": "user",
            "content":  user_message
        })

        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history[-self.max_history:]

        try:
            stream = self.client. chat.completions. create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=250,
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0]. delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text

            self.conversation_history. append({
                "role": "assistant",
                "content":  full_response
            })

        except Exception as e:
            print(f"LLM Streaming Error: {e}")
            yield "I apologize, I'm having technical difficulties."

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []

    def get_greeting(self) -> Tuple[str, dict]:
        """Get initial greeting."""
        start_time = time. perf_counter()

        greeting = """Hello and welcome to XYZ Bank!  I'm Alex, your virtual banking assistant. 
How can I help you today? """

        self.conversation_history. append({
            "role": "assistant",
            "content": greeting
        })

        return greeting, {"latency_ms": (time.perf_counter() - start_time) * 1000}


# Test module
if __name__ == "__main__":
    print("Testing LLM Module...")
    print("-" * 50)

    llm = BankingLLM(model="llama-3.1-8b-instant")

    greeting, metrics = llm.get_greeting()
    print(f"Agent:  {greeting}")
    print(f"Latency: {metrics['latency_ms']:.1f}ms")
    print("-" * 50)

    test_queries = [
        "What's your best savings account rate?",
        "I think someone stole my card",
        "How do I send money to a friend?"
    ]

    for query in test_queries:
        print(f"\nCustomer: {query}")
        response, metrics = llm.get_response(query)
        print(f"Agent: {response}")
        print(f"[Latency: {metrics['latency_ms']:.1f}ms | Tokens: {metrics. get('tokens', 'N/A')}]")