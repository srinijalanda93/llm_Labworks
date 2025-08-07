<h1>LAB-2</h1>
<H2>Implement a story generation Natural Language Generation using any transformer-based Foundation models.</H2>
Comparative Study on any three models:
• Understand the importance of Natural Language Generation and its role in NLP.
• Discuss the relevance of story generation as an NLG task.
• Compare and contrast different transformer-based models such as GPT-
2/3/Neo, BERT, T5, BART, and FLAN-T5, specifically in the context of

creative text generation.
Model Selection and Justification:
• Select a suitable pre-trained transformer model for your story generation task.
• Justify your choice based on model architecture, training corpus,generation capabilities, and efficiency.
• Optionally, consider using models from Hugging Face Transformers library.
<a href="https://colab.research.google.com/drive/13_L_-OJnWGO4l2_WRVV_2fIJvmDEg_LR?usp=sharing">GOOGLE COL IMPLEMENTATION WORK</a>
<h2>Streamlit implemented part</h2>
<img width="1470" alt="Screenshot 2025-07-08 at 10 30 31 AM" src="https://github.com/user-attachments/assets/d9e15a1a-d027-4b2e-a4cf-3e9727d3c79d" />
<img width="1470" alt="Screenshot 2025-07-08 at 10 30 39 AM" src="https://github.com/user-attachments/assets/dde52714-2b84-4cb9-ae81-cef18bd45fd0" />
<img width="1470" alt="Screenshot 2025-07-08 at 10 30 48 AM" src="https://github.com/user-attachments/assets/05cd3333-d99a-4266-a75c-98052b2d9e18" />
<img width="1470" alt="Screenshot 2025-07-08 at 10 30 57 AM" src="https://github.com/user-attachments/assets/38687511-817c-4ebc-8de6-34305e363893" />




<hr>

<h2>LAB-3</h2>
<h3>Implement a Comparative Analysis of Foundation and Domain-Specific Models</h3>
<h3>Comparative Study on any three models: (1 foundation and 2 Domain-Specific)</h3>
• Understand the importance of Domain-Specific models in various domains (Finance, Education, Healthcare, Legal, Sports, E-commerce,News & Media and more)
• Compare and contrast different transformer-based foundation models such as GPT, BERT, T5, BART, FLAN-T5 or other with domain-specificmodels Model Selection
• Domain-Specific Model Examples:
Finance: FinBERT, BloombergGPT, Healthcare: BioGPT, ClinicalBERT
Legal: Legal-BERT, CaseLaw-BERT, Education: EduBERT, SciBERT
Sports/Media: MediaSum-based T5, E-commerce: AmazonProductBERT
<h3>Evalution metrices</h3> using BOLUE ETC
<a href="https://colab.research.google.com/drive/1WvXjWli2vU5D9tvdFOGqvrmBtfcqt6Gu?usp=sharing">GOOGLE COL IMPLEMENTATION WORK</a>
<h2>Streamlit implemented part</h2>
<img width="1470" alt="Screenshot 2025-07-08 at 10 17 45 AM" src="https://github.com/user-attachments/assets/53b61a89-4e52-4c64-8f51-1ac032634f24" />
<img width="1470" alt="Screenshot 2025-07-08 at 10 17 58 AM" src="https://github.com/user-attachments/assets/1d10a32f-096b-4526-ab32-89ada922a9a1" />


<hr>
<h1>Lab-4</h1>

<img width="1470" alt="Screenshot 2025-07-10 at 11 01 08 AM" src="https://github.com/user-attachments/assets/23d3b0e4-3151-413f-8798-a072d2161837" />
<img width="1470" alt="Screenshot 2025-07-10 at 11 01 27 AM" src="https://github.com/user-attachments/assets/fe633c72-13cb-4f46-b652-7ac9fdae7209" />
<img width="1470" alt="Screenshot 2025-07-10 at 11 01 35 AM" src="https://github.com/user-attachments/assets/106dfb68-e5fd-459c-8e23-b78454347a1c" />
<img width="1470" alt="Screenshot 2025-07-10 at 11 49 17 AM" src="https://github.com/user-attachments/assets/5bd96183-b654-45ff-bc50-5f251438d80d" />




<hr>
<h1>lab 5 Sequence signal.token classification,QA-->BERT AND BERT BASED MODEL,insight,API required</h1>
<h2>Evalution sequence f1,token entity recall,qa f1 score</h2>
<img width="1470" height="790" alt="Screenshot 2025-07-24 at 11 21 58 AM" src="https://github.com/user-attachments/assets/6522f6f4-2297-4c46-ab6a-2279f8dffdcf" />


<img width="1470" height="790" alt="Screenshot 2025-07-24 at 11 22 37 AM" src="https://github.com/user-attachments/assets/a01577a1-126a-48d6-901d-e8a20055059a" />

<img width="1470" height="835" alt="Screenshot 2025-07-24 at 11 22 50 AM" src="https://github.com/user-attachments/assets/69ed6ab6-34c1-4c1c-8ffe-d3970b2508f4" />

<hr>
<h1>Lab-6 Logical Q & A system</h1>
<p>
Syllogism
Direction-based problems
Blood relation
Number series
Puzzle-based logical questions
Cause-and-effect or assumption-based MCQs
</p>
<h2>Here, provide the confidence level of the  used model since, in summary, we can use the F1 score, which regards a model fine-tuning</h2>
<h2>Use case of using confidence level: a probability, expressed as a percentage, indicating the accuracy of the model used at the prediction level. </h2>
<h2>What does F1-score the F1 score is a summary metric. where it considers Recall and precision </h2>
<table>
  <tr>
    <th>Language</th>
    <th>Model name</th>
    <th>Type of scenario</th>
    <th>Confidence-score</th>
  </tr>
  <tr>
    <th>English</th>
    <th>deepset/roberta-base-squad2</th>
    <th>Cause-and-effect or assumption-based MCQs
      "Is it correct to assume that fuel prices affect goods prices?"->The government increased taxes on fuel. The prices of all goods increased.
    </th>
    <th>0.2057</th>
  </tr>
  
</table>


<img width="995" height="776" alt="Screenshot 2025-08-07 at 11 59 23 AM" src="https://github.com/user-attachments/assets/487977bb-87b4-4028-a28a-a7d0008b73d3" />
<img width="706" height="793" alt="Screenshot 2025-08-07 at 12 02 19 PM" src="https://github.com/user-attachments/assets/ea15b3b9-e7b8-4831-9910-a622d7dfae43" />



