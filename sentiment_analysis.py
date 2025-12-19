import pandas as pd
import numpy as np
import re, time, os
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import lexicon_fix_ai

def load_data(base_dir):
    ##load data and convert date column
    headlines_df = pd.read_csv(os.path.join(base_dir, 'sp500_headlines_2008_2024.csv'))
    lexicon_df = pd.read_csv(os.path.join(base_dir, 'financial_sentiment_lexicon.csv'))
    headlines_df['Date'] = pd.to_datetime(headlines_df['Date'])

    ##fix lexicon using domain knowledge to correct inconsistent signs - AI ADDITION!
    lexicon_df = lexicon_fix_ai.apply_lexicon_corrections(lexicon_df)

    ##create lexicon dictionary
    lexicon_dict = {str(k).lower(): v for k, v in zip(lexicon_df['Word_or_Phrase'], lexicon_df['Sentiment_Score'])}
    
    ##calculate max phrase length for matching of multi-word terms
    max_phrase_len = max([len(str(k).split()) for k in lexicon_dict.keys()]) if lexicon_dict else 1
    
    return headlines_df, lexicon_dict, max_phrase_len

def get_lexicon_sentiment(title, lexicon_dict, max_phrase_len):
    if pd.isna(title): return 0.0
    words = re.findall(r'\b\w+\b', str(title).lower())
    score, count, i, n = 0.0, 0, 0, len(words)
    
    while i < n:
        ##try to match longest phrase first -> greedy approach
        match_found = False
        for length in range(max_phrase_len, 0, -1):
            phrase = " ".join(words[i : i + length])
            if phrase in lexicon_dict:
                score += lexicon_dict[phrase]
                count += 1
                i += length
                match_found = True
                break
        if not match_found: i += 1
            
    return score / count if count > 0 else 0.0

##calculate accuracy metrics helper
def get_metrics(df, s_col, t_col):
    if df.empty: return 0.0, 0.0
    # mimics the original lazy logic so results match exactly
    acc = np.mean((df[s_col] > 0) == (df[t_col] > 0))
    try: 
        corr = np.nan_to_num(np.corrcoef(df[s_col], df[t_col])[0, 1])
    except: 
        corr = 0.0
    return acc, corr

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    headlines_df, lexicon_dict, max_phrase_len = load_data(base_dir)

    ##prepare data - record linking and next-day changes
    daily_prices = headlines_df[['Date', 'CP']].drop_duplicates().sort_values('Date')
    daily_prices['price_change_pct'] = daily_prices['CP'].pct_change().shift(-1) * 100
    daily_prices = daily_prices.dropna(subset=['price_change_pct'])
    sample_data = headlines_df.copy() 

    ##lexicon analysis
    start_time = time.time()
    sample_data['lexicon_sentiment'] = sample_data['Title'].apply(lambda x: get_lexicon_sentiment(x, lexicon_dict, max_phrase_len))
    lexicon_avg_time = ((time.time() - start_time) / len(sample_data) * 1000) if len(sample_data) > 0 else 0

    ##Transformer analysis
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    pipe = pipeline("sentiment-analysis", model=AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert"), tokenizer=tokenizer)
    
    start_time = time.time()
    ##process all headlines in batch
    results = pipe(sample_data['Title'].astype(str).tolist(), truncation=True, padding=True)
    sample_data['transformer_sentiment'] = [
        r['score'] if r['label'] == 'positive' else -r['score'] if r['label'] == 'negative' else 0 
        for r in results
    ]
    advanced_avg_time = ((time.time() - start_time) / len(sample_data) * 1000) if len(sample_data) > 0 else 0

    ##Aggregation Logic DAILY & WEEKLY
    daily_sentiment = sample_data.groupby('Date')[['lexicon_sentiment', 'transformer_sentiment']].mean().reset_index()
    analysis_df = daily_sentiment.merge(daily_prices[['Date', 'price_change_pct']], on='Date')
    
    weekly_sentiment = daily_sentiment.set_index('Date').resample('W-FRI')[['lexicon_sentiment', 'transformer_sentiment']].mean()
    ##Resample prices to get Friday Close and calc return
    weekly_returns = headlines_df[['Date', 'CP']].drop_duplicates().sort_values('Date').set_index('Date')['CP'].resample('W-FRI').last().pct_change().shift(-1) * 100
    weekly_df = pd.concat([weekly_sentiment, weekly_returns.rename('weekly_return')], axis=1).dropna()

    ##Metrics Calculation
    lex_acc_d, lex_corr_d = get_metrics(analysis_df, 'lexicon_sentiment', 'price_change_pct')
    adv_acc_d, adv_corr_d = get_metrics(analysis_df, 'transformer_sentiment', 'price_change_pct')
    lex_acc_w, lex_corr_w = get_metrics(weekly_df, 'lexicon_sentiment', 'weekly_return')
    adv_acc_w, adv_corr_w = get_metrics(weekly_df, 'transformer_sentiment', 'weekly_return')

    print(f"\n DAILY RESULTS\nLexicon Accuracy: {lex_acc_d:.2%} | Corr: {lex_corr_d:.3f}\nTransformer Accuracy: {adv_acc_d:.2%} | Corr: {adv_corr_d:.3f}")
    print(f"\nWEEKLY RESULTS \nLexicon Accuracy: {lex_acc_w:.2%} | Corr: {lex_corr_w:.3f}\nTransformer Accuracy: {adv_acc_w:.2%} | Corr: {adv_corr_w:.3f}")
    print(f"\nComparison:\nLexicon Speed: {lexicon_avg_time:.2f}ms\nTransformer Speed: {advanced_avg_time:.2f}ms")

    ##VISUALIZATION BLOCK
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ##Plot 1 Accuracy
    x = np.arange(2)
    rects1 = axes[0].bar(x - 0.35/2, [lex_acc_d, adv_acc_d], 0.35, label='Daily', color='skyblue')
    rects2 = axes[0].bar(x + 0.35/2, [lex_acc_w, adv_acc_w], 0.35, label='Weekly', color='navy')
    axes[0].set(ylim=(0, 1.0), title='Accuracy: Daily vs Weekly', ylabel='Accuracy', xticks=x, xticklabels=['Lexicon', 'Transformer'])
    axes[0].legend()
    for r in rects1 + rects2: axes[0].annotate(f'{r.get_height():.1%}', (r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    ##Plot 2 Latency
    axes[1].bar(['Lexicon', 'Transformer'], [lexicon_avg_time, advanced_avg_time], color=['gray', 'orange'])
    axes[1].set(yscale='log', title='Latency per Headline (Log Scale)', ylabel='Milliseconds (ms)')
    for i, v in enumerate([lexicon_avg_time, advanced_avg_time]): axes[1].text(i, v * 1.1, f'{v:.2f} ms', ha='center')

    ##Plot 3 Correlation
    colors = np.where(np.sign(analysis_df['transformer_sentiment']) == np.sign(analysis_df['price_change_pct']), 'green', 'red')
    axes[2].scatter(analysis_df['transformer_sentiment'], analysis_df['price_change_pct'], c=colors, alpha=0.5)
    axes[2].set(title=f'Transformer Daily Accuracy\nCorr: {adv_corr_d:.3f}', xlabel='Daily Sentiment Score', ylabel='Daily Price Change (%)')
    axes[2].axhline(0, color='black', lw=0.8); axes[2].axvline(0, color='black', lw=0.8)

    ##text, visual aid etc
    props = dict(facecolor='white', alpha=0.5)
    axes[2].text(0.9, 0.9, 'True Positive\n(Green)', ha='right', va='top', transform=axes[2].transAxes, fontsize=9, alpha=0.8, bbox=props)
    axes[2].text(0.1, 0.9, 'False Negative\n(Red)', ha='left', va='top', transform=axes[2].transAxes, fontsize=9, alpha=0.8, bbox=props)
    axes[2].text(0.1, 0.1, 'True Negative\n(Green)', ha='left', va='bottom', transform=axes[2].transAxes, fontsize=9, alpha=0.8, bbox=props)
    axes[2].text(0.9, 0.1, 'False Positive\n(Red)', ha='right', va='bottom', transform=axes[2].transAxes, fontsize=9, alpha=0.8, bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'results_dashboard.png'))

    ##for model_disagreements.csv
    disagreements = analysis_df[(np.sign(analysis_df['lexicon_sentiment']) != np.sign(analysis_df['transformer_sentiment'])) & (analysis_df['lexicon_sentiment'] != 0) & (analysis_df['transformer_sentiment'] != 0)].copy()
    
    if not disagreements.empty:
        disagreements['Lexicon_Correct'] = np.sign(disagreements['lexicon_sentiment']) == np.sign(disagreements['price_change_pct'])
        disagreements['Transformer_Correct'] = np.sign(disagreements['transformer_sentiment']) == np.sign(disagreements['price_change_pct'])
        
        interesting = sample_data[sample_data['Date'].isin(disagreements['Date'])]
        if not interesting.empty:
            interesting.to_csv(os.path.join(base_dir, 'model_disagreements.csv'), index=False)

    ##save full results in same path
    analysis_df.to_csv(os.path.join(base_dir, 'sentiment_analysis_results.csv'), index=False)
    return {'lexicon_speed_ms': lexicon_avg_time, 'lexicon_accuracy': lex_acc_d, 'advanced_speed_ms': advanced_avg_time, 'advanced_accuracy': adv_acc_d}

if __name__ == "__main__":
    results = main()
