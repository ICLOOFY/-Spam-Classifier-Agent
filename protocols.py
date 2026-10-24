'''Spam Classifier Agent - schemas, threat dictionaries, and the deterministic
red-flag keyword scoring engine (Principal Email Security Analyst persona).'''

import re
from datetime import datetime
from uuid import uuid4

from uagents import Context, Field, Model, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)


# ---------- Message Schemas ----------
class MessageToClassify(Model):
    text: str = Field(description='Raw email or message body to classify')
    sender_hint: str = Field(
        default='unknown',
        description='Optional display hint, NOT used for scoring',
    )


class ClassificationResult(Model):
    classification: str = Field(description='The verdict: SPAM or HAM')
    confidence: float = Field(description='Confidence in the verdict, 0.0 to 1.0')
    detected_red_flags: list[str] = Field(
        description='Red-flag phrases found, formatted like: click here x2'
    )
    routing_action: str = Field(description='Where to route the message: INBOX or JUNK_FOLDER')
    reasoning: str = Field(description='Analyst-style explanation of the decision')


# ---------- Threat Vector Dictionaries (evidence-grounded) ----------
FINANCIAL_PROMOTION = [
    'free', 'win', 'winner', 'prize', 'guaranteed',
    'click here', 'gift card', 'cash', 'claim your', 'no cost',
    'act now', 'limited offer', 'exclusive deal',
]

URGENCY_ACCOUNT_THREAT = [
    'urgent', 'immediate action required', 'account suspended',
    'final notice', 'verify your account', 'account locked',
    'password expires', 'unauthorized login', 'security alert',
    'confirm your identity', 'update payment',
]

SPAM_THRESHOLD = 3  # weighted score at or above this is classified SPAM

# ---------- Anti-evasion normalizer ----------
# Map leet digits back to letters; symbol masks are stripped by the regex below.
_MASK_STRIP = str.maketrans({'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't'})
# Additionally map symbol masks to letters so W!N, FR33 or ca$h still resolve.
_LEET_MAP = str.maketrans({
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't',
    '$': 's', '@': 'a', '!': 'i', '|': 'l', '+': 't',
})


def normalize(text: str) -> tuple[str, str]:
    '''Return two matching variants of the input:
    collapsed - mask characters stripped, so (act now!) still matches (act now);
    de_leeted - leet symbols mapped to letters, so W!N or FR33 match win/free'''
    lowered = text.lower()
    collapsed = re.sub(r'[^a-z0-9]+', ' ', lowered.translate(_MASK_STRIP)).strip()
    de_leeted = re.sub(r'[^a-z0-9]+', ' ', lowered.translate(_LEET_MAP)).strip()
    return collapsed, de_leeted


def _count_hits(keyword: str, variants: tuple[str, str]) -> int:
    '''Whole-word/phrase occurrence count; takes the best variant so masked
    spellings count once, and win never matches inside window.'''
    pattern = r'(?<![a-z0-9])' + re.escape(keyword) + r'(?![a-z0-9])'
    return max(len(re.findall(pattern, v)) for v in variants)


# ---------- Core Classification (deterministic, temp ~0.1) ----------
def classify(text: str) -> ClassificationResult:
    '''Deterministic red-flag frequency analysis: the same input always
    produces the same SPAM/HAM verdict, confidence and reasoning.'''
    collapsed, de_leeted = normalize(text)
    if not collapsed:
        return ClassificationResult(
            classification='HAM',
            confidence=0.9,
            detected_red_flags=[],
            routing_action='INBOX',
            reasoning='No analyzable content was provided; defaulting to HAM.',
        )

    detected: list[str] = []
    cat_hits = {'financial promotion': 0, 'urgency/account threat': 0}
    score = 0
    total_hits = 0

    for category, keywords in (
        ('financial promotion', FINANCIAL_PROMOTION),
        ('urgency/account threat', URGENCY_ACCOUNT_THREAT),
    ):
        for keyword in keywords:
            hits = _count_hits(keyword, (collapsed, de_leeted))
            if hits:
                # Multi-word threat phrases weigh double.
                score += hits * (2 if ' ' in keyword else 1)
                total_hits += hits
                cat_hits[category] += hits
                detected.append(f'{keyword} x{hits}')

    if score >= SPAM_THRESHOLD:
        fin = cat_hits['financial promotion']
        urg = cat_hits['urgency/account threat']
        return ClassificationResult(
            classification='SPAM',
            confidence=min(0.99, round(0.60 + 0.07 * score, 2)),
            detected_red_flags=detected,
            routing_action='JUNK_FOLDER',
            reasoning=(
                f'Detected {total_hits} red-flag occurrence(s) across '
                f'{len(detected)} distinct phrase(s) ({fin} financial/promotional, '
                f'{urg} urgency/account-threat); weighted score {score} meets or '
                f'exceeds the spam threshold of {SPAM_THRESHOLD}.'
            ),
        )

    return ClassificationResult(
        classification='HAM',
        confidence=max(0.5, round(0.95 - 0.15 * score, 2)),
        detected_red_flags=detected,
        routing_action='INBOX',
        reasoning=(
            f'Weighted red-flag score {score} is below the spam threshold of '
            f'{SPAM_THRESHOLD}; no strong spam or phishing pattern was detected.'
        ),
    )


def format_result(result: ClassificationResult, sender_hint: str = 'unknown') -> str:
    icon = '🚨' if result.classification == 'SPAM' else '✅'
    flags = ', '.join(result.detected_red_flags) if result.detected_red_flags else 'none'
    hint = '' if sender_hint == 'unknown' else f' (from: {sender_hint})'
    return (
        f'{icon} Email Security Analysis{hint}\n\n'
        f'Classification: {result.classification}\n'
        f'Confidence: {result.confidence}\n'
        f'Routing action: {result.routing_action}\n'
        f'Detected red flags: {flags}\n\n'
        f'Reasoning: {result.reasoning}'
    )


def create_text_chat(text: str) -> ChatMessage:
    content = [TextContent(type='text', text=text)]
    return ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=content)


# ---------- Chat Protocol (Agentverse chat UI) ----------
chat_protocol = Protocol(spec=chat_protocol_spec)


@chat_protocol.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )
    text = msg.text()
    if not text:
        return
    await ctx.send(sender, create_text_chat(format_result(classify(text))))


@chat_protocol.on_message(ChatAcknowledgement)
async def handle_chat_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


# ---------- Structured Agent-to-Agent Protocol ----------
classifier_protocol = Protocol('SpamClassification')


@classifier_protocol.on_message(model=MessageToClassify, replies=ClassificationResult)
async def handle_classify(ctx: Context, sender: str, msg: MessageToClassify):
    ctx.logger.info(
        f'Classifying message from hint {msg.sender_hint!r} ({len(msg.text)} chars)'
    )
    await ctx.send(sender, classify(msg.text))
