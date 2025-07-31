# Text_Summarization_Tool.py
from transformers import pipeline

def summarize_text(text, max_length=150, min_length=30):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def main():
    sample_text = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "unlike the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of \"intelligent agents\": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. "
        "Colloquially, the term \"artificial intelligence\" is often used to describe machines (or computers) that mimic \"cognitive\" functions that humans associate with the human mind, such as \"learning\" and \"problem solving\"."
    )

    print("Original Text:")
    print(sample_text)
    print("\nSummary:")
    print(summarize_text(sample_text))

if __name__ == "__main__":
    main()
