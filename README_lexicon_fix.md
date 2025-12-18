# Lexicon Fix AI Module

This document explains the changes made by the AI to correct the `financial_sentiment_lexicon.csv` used in the sentiment analysis project.

## The Problem
The original lexicon contained significant errors where the sentiment score signs were inverted:
- **False Negatives**: Positive terms had negative scores (e.g., "stock chart", "economic expansion").
- **False Positives**: Clearly negative terms had positive scores (e.g., "corruption probe", "cyber attack").

## The Solution
A new module `lexicon_fix_ai.py` was created to programmatically correct these errors at runtime without permanently altering the source CSV file (though it can be used to generate a fixed CSV).

### Corrections Applied

#### 1. Misclassified Positive Terms (Sign Flipped to Positive)
The following terms, previously negative, are now correctly scored as **Positive**:
- `stock chart`
- `technical analysis`
- `fundamental analysis`
- `breakout pattern`
- `double bottom`
- `trend line`
- `momentum`
- `risk appetite`
- `economic expansion`
- `quantitative easing`

#### 2. Misclassified Negative Terms (Sign Flipped to Negative)
The following terms, previously positive, are now correctly scored as **Negative**:
- `antitrust scrutiny`
- `compliance violations`
- `corruption probe`
- `bribery accusations`
- `money laundering charges`
- `fraud investigation`
- `insider trading scandal`
- `SEC enforcement action`
- `class-action lawsuit`
- `patent infringement`
- `cyber attack`
- `ransomware incident`
- `network outage`
- `system failure`
- `reputation risk`
- `brand damage`
- `inventory shortage`
- `shipping delays`
- `logistics bottlenecks`
- `raw material shortages`
- `rising input costs`

#### 3. Misclassified Neutral Terms (Score neutralized to 0.0)
The following terms, previously polarized (often negative), are now correctly marked as **Neutral (0.0)** to prevent noise:
- `market`, `trading`, `investing`, `earnings`, `dividend`
- `merger`, `acquisition`, `regulation`, `public relations`
- `volatility`, `guidance`, `IPO`, `split`, `buyback`

## Usage
The `lexicon_fix_ai.py` module exposes a function `apply_lexicon_corrections(lexicon_df)` which takes the pandas DataFrame of the lexicon and returns the corrected version. It ensures:
1.  **Positive List**: Score > 0
2.  **Negative List**: Score < 0
3.  **Neutral List**: Score = 0

