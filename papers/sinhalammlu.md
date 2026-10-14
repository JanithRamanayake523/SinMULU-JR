# SinhalaMMLU: A Comprehensive Benchmark for Evaluating Multitask Language Understanding in Sinhala

**Ashmari Pramodya**¹, **Nirasha Nelki**², **Heshan Shalinda**³, **Chamila Liyanage**², **Yusuke Sakai**¹, **Randil Pushpananda**², **Ruvan Weerasinghe**³, **Hidetaka Kamigaito**¹, **Taro Watanabe**¹

¹ Nara Institute of Science and Technology (NAIST), Japan  
² University of Colombo School of Computing (UCSC), Sri Lanka  
³ Informatics Institute of Technology (IIT), Sri Lanka  

`{pussewala.ashmari.ow4, sakai.yusuke.sr9, kamigaito.h, taro}@is.naist.jp`  
`{2021cs127@stu, rpn@, cml@}ucsc.cmb.ac.lk`, `{heshan.20230591,ruvan.w}@iit.ac.lk`

---

## Abstract

Large Language Models (LLMs) demonstrate impressive general knowledge and reasoning abilities, yet their evaluation has predominantly focused on global or anglocentric subjects, often neglecting low-resource languages and culturally specific content. While recent multilingual benchmarks attempt to bridge this gap, many rely on automatic translation, which can introduce errors and misrepresent the original cultural context. To address this, we introduce **SinhalaMMLU**, the first multiple-choice question answering benchmark designed specifically for Sinhala, a low-resource language. The dataset includes over 7,000 questions spanning secondary to collegiate education levels and is aligned with the Sri Lankan national curriculum. It covers six domains and 30 subjects, encompassing both general academic topics and culturally grounded knowledge. We evaluate 26 LLMs on SinhalaMMLU and observe that, while Claude 3.5 Sonnet and GPT-4o achieve the highest average accuracies at 67% and 62% respectively, overall model performance remains limited. In particular, models struggle in culturally rich domains such as the Humanities, revealing substantial room for improvement in adapting LLMs to low-resource and culturally specific contexts.

---

## 1 Introduction

The introduction of large language models (LLMs) has led to unprecedented improvements in natural language processing (NLP) and artificial intelligence, reshaping the field with their revolutionary capabilities (OpenAI et al., 2024; Anthropic, 2024; Touvron et al., 2023). Despite multilingual capabilities in LLMs, significant performance gaps persist between English and other languages (Liu et al., 2024). This performance gap is particularly pronounced for languages using non-Latin scripts such as Sinhala. One critical factor is the lack of high-quality benchmarks for non-English languages, particularly those that are low-resourced. The majority of current evaluation frameworks remain English-centric (Hendrycks et al., 2021), creating significant assessment gaps for linguistically diverse and low-resource languages. Many benchmarks claiming multilingual support are simply translated versions (Bandarkar et al., 2024; Singh et al., 2025) of content originally designed for English speakers. These often include content that assumes knowledge of the American legal system or requires familiarity with English-specific cultural references and colloquialisms. Despite appearing multilingual through translation, these benchmarks fail to reflect the cultural context (Ji et al., 2023) valued by native speakers.

To address the lack of existing benchmarks for the Sinhala NLP community and to better assess how LLMs perform on Sinhala, we introduce **SinhalaMMLU**, a comprehensive evaluation framework comprising over 7,000 questions across 6 domains and 30 subjects reflecting both general and culturally specific knowledge. SinhalaMMLU is designed by collecting questions from various national and provincial exams, without relying on translations. These questions include culturally relevant subjects such as Sri Lankan history, arts, drama and theatre, indigenous dancing, oriental music, and Sinhala language, alongside traditional academic subjects like science.

We evaluate 26 different LLMs, a mix of closed proprietary and open-source. Our results show significant room for improvement, with Claude 3.5 Sonnet scoring the highest at 67.65%. We also explore how performance scales with the number of parameters, finding significant improvements as model size increases. Our domain-wise analysis reveals that models underperform in culturally relevant areas, such as Humanities and Language, as well as in general subjects within the STEM domain. All dataset details are publicly available on [GitHub](https://github.com/naist-nlp/SinhalaMMLU) and [Hugging Face](https://huggingface.co/datasets/naist-nlp/SinhalaMMLU).

---

## 2 Related Work

### 2.1 Benchmarks for LLMs
In an era where different LLMs are emerging day by day, benchmarks are crucial to understanding their power and performance. Benchmarks like GLUE (Wang et al., 2018), SuperGLUE (Wang et al., 2019), and SQuAD (Rajpurkar et al., 2016) are being used as evaluation frameworks for natural language understanding (NLU) tasks, including reading comprehension and question-answering. Eventually, multilingual evaluation frameworks emerged like XGLUE (Liang et al., 2020) and XTREME-R (Ruder et al., 2021). However, these benchmarks only assess the linguistic skills of the language rather than language understanding.

MMLU (Hendrycks et al., 2021) is one of the prominently used benchmarks for LLMs. It evaluates the LLMs via multiple-choice questions belonging to different subjects and domains to evaluate the knowledge acquisition of LLMs. Since MMLU consists of English questions, there are some attempts to create similar benchmarks for other languages, including Arabic (Koto et al., 2024), Chinese (Li et al., 2024), Turkish (Yüksel et al., 2024), Indonesian (Koto et al., 2023), Korean (Son et al., 2025), and Persian (Ghahroodi et al., 2024). Some MMLU benchmarks have been constructed for low-resourced languages, notably African (Adelani et al., 2025), Malay (Poh et al., 2024), and Basque (Etxaniz et al., 2024). The MILU benchmark, which consists of 11 Indic languages, stands out among the language-specific MMLU initiatives (Verma et al., 2025). In an effort to establish a multilingual benchmark, the INCLUDE benchmark (Romanou et al., 2024) has achieved a remarkable feat as a benchmark by creating an impressive QA dataset that encompasses a variety of exams from various languages.

### 2.2 Sinhala Language
Sinhala is an Indo-Aryan language spoken by over 17 million people in Sri Lanka, where it serves as one of the two official languages. Despite its sizable community, Sinhala remains relatively under-resourced in terms of language technology and NLP tools, though there have been some attempts for it.

The existing textual datasets for Sinhala have been curated for different purposes like text summarization (Hasan et al., 2021; Hewapathirana et al., 2024) and text classification (Hettiarachchi et al., 2024; Warusawithana et al., 2022; Ranasinghe et al., 2024). The Aya Dataset (Singh et al., 2024), a human-annotated multilingual instruction fine-tuning dataset, also includes Sinhala data. However, these works have only focused on a single task or a single domain; therefore, they are not varied enough to consider for a multi-task benchmark for LLMs. Since the MMLU has been proposed for English, there have been attempts to create multilingual benchmarks by translating the initial dataset into other languages, including Sinhala (Singh et al., 2025). However, there is a question regarding the accuracy of the translation, as it can be challenging to translate language-specific terms. A language-specific benchmark or a multilingual benchmark should consider culture-related aspects as well, in order to ensure LLMs understand the cultural contexts that are lacking in GlobalMMLU (Singh et al., 2025).

---

## 3 SinhalaMMLU

Motivated by the scarcity of high-quality resources for Sinhala, we introduce **SinhalaMMLU**, a benchmark designed to evaluate language models in the Sinhala language. The benchmark includes multiple-choice questions contextually designed for the Sinhala language, ranging over a wide range of subjects and educational levels. Following the structure and methodology of the original English MMLU benchmark, we carefully curated the dataset in alignment with the local national curriculum to ensure both relevance and cultural appropriateness.

The national educational curriculum of Sri Lanka is divided into three broad levels: primary, secondary, and collegiate education. Primary education spans five years (Grades 1-5) for children aged 5 to 9, focusing on foundational subjects. However, we excluded this level from SinhalaMMLU, as the questions are generally simple (e.g., drawing, filling blanks, or writing) and are not suited for multiple-choice evaluation. Secondary education is divided into two phases: junior secondary (Grades 6 to 9, ages 10 to 13) and senior secondary (Grades 10 to 11, ages 14 to 16), culminating in the General Certificate of Education Ordinary Level (GCE O-Level) examination, where students take six main subjects and three additional subjects. The collegiate level comprises Grades 12-13 (ages 17-19) and concludes with the GCE Advanced Level (A-Level) examination. There are five academic streams at this level: Arts, Commerce, Biological Science, Physical Science, and Technology. Students select at least three subjects within their chosen stream, and the national examination includes these core subjects along with a General English test and a Common General Test. The A-Level exam serves as the primary qualification for university entrance in Sri Lanka.

All questions were prepared in alignment with the official curriculum set by the Ministry of Education (MOE), Sri Lanka. By aligning SinhalaMMLU with educational standards, we aim to provide a rigorous and contextually grounded benchmark to evaluate the performance of LLMs in Sinhala across educational levels. This allows us to systematically assess model understanding of local knowledge, identify linguistic or subject-specific weaknesses, and encourage the development of culturally relevant and educationally aligned language models for low-resource languages.

### 3.1 Data Preparation
Our primary data source was *e-Thaksalawa*, the public official e-learning platform developed by the Sri Lankan MOE. This platform provides free access to a wide range of educational materials, including past exam papers, lecture notes, interactive textbooks, video lessons, and multiple-choice questions (MCQs) with answer keys curated by domain experts. We collected MCQs aligned with the national curriculum from government-focused papers, specifically from the GCE O-Level, GCE A-Level, and provincial examination papers for Grades 6-8. These exams include multiple-choice sections covering various academic subjects. In addition to e-Thaksalawa, we used other educational websites that host government-released past papers and marking schemes.

To curate the data, we employed four annotators with undergraduate or higher educational qualifications. Over a two-month period, the annotators manually extracted MCQs from PDF documents (after OCR), while quiz-based content was scraped directly from the web platform. Annotators were instructed to include only questions that were in multiple-choice format with clearly defined answer options and an identified correct answer. Questions containing multimodal content, such as images, audio, or video, were excluded. Furthermore, we excluded Mathematics questions because they primarily involve symbolic reasoning and are not usually provided as MCQ questions. More than 65% of the collected data was sourced from PDFs.

### 3.2 Data Formatting
Each MCQ in SinhalaMMLU follows a consistent structure composed of a question stem, a set of four or five choices, and a single correct answer. The dataset includes both fill-in-the-blank style questions and direct queries. Along with each question, we recorded metadata such as the subject, difficulty level mapped to school grade, original source and year, and the province, especially in the case of government-issued or provincial papers. For chemical formulae and mathematical expressions, we use a 50:50 mixture of $\text{\LaTeX}$ and plain text, where plain text was only allowed if an expression is commonly used and not prone to ambiguity. To ensure content uniqueness and eliminate redundancy, we removed exact duplicate questions with identical answer sets and applied cosine similarity analysis. Questions with a similarity score exceeding 95% were flagged and removed from the dataset.

### 3.3 Difficulty Classification
We classified question difficulty as follows:
- **Easy**: Questions from Grades 6-8. Most of these had four answer choices, although a subset of geography questions (44 in total) included only three. To maintain format consistency, we added a fourth option using Claude 3.5 Sonnet, which was not used during model evaluation with our dataset.
- **Medium**: Questions from Grades 9-11, all of which contained four answer options.
- **Hard**: Questions from Grades 12-13, typically featuring five answer choices.

This classification mirrors the increasing academic depth and complexity expected at higher grade levels and provides a clear basis for evaluating model performance across educational stages.

### 3.4 Data Distribution
Table 1 summarizes the dataset statistics, including the average question length across 7,044 questions. Each subject has at least 104 questions, which we split into a few-shot set with 3 questions, and a test set with more than 100 questions. As shown in Table 1 and Table 2, SinhalaMMLU includes the following domains: **Humanities** (which is deeply rooted in Sri Lankan cultural and historical context), **STEM**, **Social Science**, **Business Studies**, **Language**, and **Other**.

| Group / Domain | # Questions | Question Avg Length (Chars) | Answer Avg Length (Chars) |
| :--- | :--- | :--- | :--- |
| **Easy** | 1893 | 59.08 | 16.77 |
| **Medium** | 2585 | 100.66 | 24.79 |
| **Hard** | 2566 | 116.40 | 27.53 |
| **STEM** | 629 | 27.42 | 157.82 |
| **Social Science** | 1084 | 141.80 | 22.34 |
| **Humanities** | 3419 | 93.91 | 22.24 |
| **Language** | 397 | 74.19 | 25.65 |
| **Business Studies** | 477 | 173.39 | 32.99 |
| **Other** | 1038 | 108.58 | 28.24 |

*Table 1: Total number of questions, Average question and answer length (in characters) for each difficulty level and domain. The overall question count is 7,044.*

| Domain | Subjects |
| :--- | :--- |
| **Humanities** | History, Drama and Theatre, Dancing, Eastern Music, Arts, Buddhism, Catholicism, Christianity, Islam, Buddhist Civilization, Oriental Music, History of Sri Lanka, Dancing Indigenous |
| **Social Science** | Citizenship Education, Health and Physical Science, Geography, Political Science |
| **STEM** | Physics, Chemistry, Biology, Science |
| **Language** | Sinhala Language and Literature |
| **Business Studies** | Business and Accounting Studies, Entrepreneurship Studies, Economics |
| **Other** | Home Economics, Biosystems Technology, Communication and Media Studies, Design and Construction Technology, Agriculture and Food Technology |

*Table 2: Subjects categorized by domain in the SinhalaMMLU dataset.*

---

## 4 Experimental Setup

**Models:** We evaluated 26 recent state-of-the-art multilingual LLMs of different sizes in zero-shot and few-shot settings. These include both small and large open models as well as closed models. For open-source models, we include Cohere4AI's Aya Expanse, LLaMA-3 (Touvron et al., 2023), Qwen (Qwen et al., 2025), and Mistral (Jiang et al., 2024), along with their chat versions. To optimize memory efficiency during inference, we applied 4-bit NF4 quantization with double quantization and bfloat16 computation. For closed models, we used GPT-4o (OpenAI et al., 2024) and Claude 3.5 Sonnet / Haiku (Anthropic, 2024).

**Evaluation:** We evaluated LLMs by accuracy. Following previous studies (Koto et al., 2023; Li et al., 2024), for open-source models, we determine the answer based on the highest probability among all possible options. For closed-source models, we determine the answer based on the first token generated in the text using a regular expression.

**Prompt:** For the SinhalaMMLU evaluation, we used two distinct prompt templates: one that specifies the subject domain and another that omits this information. Both follow the instruction format proposed by Hendrycks et al. (2021), with prompt instructions given in the same language as the question.

---

## 5 Results and Discussions

This section presents the evaluation results on the SinhalaMMLU dataset. To assess the impact of subject-specific information in prompts, we compared two variants: one incorporating the subject name and one without. We observed that including the subject name generally improves performance across models. Therefore, for all subsequent experiments, we report results using prompts that include subject information.

### Results by Model
Table 3 summarizes the full zero-shot results of all the models, grouped by domain. Closed models, notably Claude 3.5 Sonnet and GPT-4o, exhibit superior performance across all domains, achieving average accuracies of 67.65% and 62.95%, respectively. These models consistently outperform open-source counterparts, indicating a higher level of generalization and understanding.

Among open-source models, Qwen2.5-72B-chat and Qwen2.5-32B-chat stand out, with average accuracies of 41.18% and 36.47%, respectively. Their performance suggests that larger model sizes and instruction tuning contribute positively to handling diverse and complex tasks. Within the LLaMA-3 model family, LLaMA-3.1-70B-chat is the only variant that achieves comparatively higher performance. In contrast, smaller models like LLaMA-3.2-1B and Mistral-7B variants show lower performance, with average accuracies hovering around 22%. This disparity underscores the importance of model scale and training data diversity in achieving higher accuracy. Scaling trends indicate that larger models systematically outperform smaller ones.

| Model | Humanities | Language | Social Sci. | STEM | Business | Other | Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLAUDE-3-5-SONNET** | **66.15** | **62.37** | **77.55** | **61.40** | **73.12** | **65.58** | **67.65** |
| CLAUDE-3-HAIKU | 41.01 | 43.81 | 50.57 | 35.34 | 44.09 | 39.64 | 42.14 |
| **GPT-4o** | 62.02 | 51.29 | 71.32 | 60.59 | 67.53 | 61.44 | 62.95 |
| QWEN2.5-7B | 22.27 | 21.65 | 25.66 | 18.57 | 20.22 | 21.70 | 22.20 |
| QWEN2.5-7B-CHAT | 28.02 | 25.00 | 28.30 | 28.50 | 27.10 | 23.67 | 27.23 |
| QWEN2.5-32B | 28.85 | 28.35 | 35.28 | 26.38 | 25.16 | 27.51 | 29.15 |
| QWEN2.5-32B-CHAT | 36.70 | 32.73 | 38.96 | 38.60 | 34.84 | 34.02 | 36.47 |
| QWEN2.5-72B | 34.69 | 28.87 | 40.57 | 37.95 | 28.82 | 34.52 | 35.14 |
| QWEN2.5-72B-CHAT | 39.84 | 38.92 | 45.85 | 45.60 | 41.29 | 38.86 | 41.18 |
| LLAMA-3.2-1B | 22.18 | 21.65 | 25.66 | 18.40 | 19.78 | 21.70 | 22.12 |
| LLAMA-3.2-1B-CHAT | 22.27 | 21.65 | 25.75 | 18.73 | 19.78 | 21.70 | 22.20 |
| LLAMA-3.2-3B | 22.18 | 22.68 | 25.57 | 18.40 | 19.57 | 21.70 | 22.14 |
| LLAMA-3.2-3B-CHAT | 22.27 | 21.65 | 25.66 | 18.57 | 19.35 | 21.70 | 22.14 |
| LLAMA-3-8B | 22.48 | 21.65 | 26.32 | 19.54 | 20.86 | 21.50 | 22.51 |
| LLAMA-3-8B-CHAT | 22.87 | 21.65 | 26.98 | 19.38 | 19.78 | 23.18 | 22.96 |
| LLAMA-3-70B | 22.48 | 23.45 | 24.95 | 22.48 | 19.78 | 22.09 | 22.65 |
| LLAMA-3-70B-CHAT | 27.15 | 20.36 | 26.32 | 23.78 | 20.65 | 22.49 | 25.21 |
| LLAMA-3.1-8B | 22.90 | 23.45 | 27.17 | 19.54 | 23.44 | 23.37 | 23.39 |
| LLAMA-3.1-8B-CHAT | 25.29 | 25.52 | 29.06 | 22.31 | 21.08 | 24.95 | 25.28 |
| LLAMA-3.1-70B | 22.19 | 22.94 | 26.42 | 19.54 | 19.78 | 21.60 | 22.40 |
| LLAMA-3.1-70B-CHAT | 27.24 | 26.80 | 25.94 | 23.78 | 26.67 | 25.25 | 26.37 |
| LLAMA-3.3-70B-CHAT | 24.54 | 23.45 | 27.26 | 23.13 | 25.81 | 22.88 | 24.61 |
| MISTRAL-7B-CHAT | 21.55 | 22.94 | 26.92 | 22.36 | 20.00 | 20.81 | 22.28 |
| MISTRAL-7B | 22.18 | 21.65 | 25.66 | 18.40 | 19.78 | 21.65 | 22.12 |
| AYA-EXPANSE-8B | 22.78 | 24.23 | 25.09 | 19.71 | 20.43 | 21.70 | 22.62 |
| AYA-EXPANSE-32B | 23.97 | 25.00 | 30.00 | 25.08 | 23.66 | 24.65 | 25.14 |
| **Average** | **27.37** | **26.10** | **31.79** | **27.16** | **27.78** | **28.07** | **28.38** |

*Table 3: Zero-shot performance (% accuracy) of LLMs across the six domains.*

### Results by Domain
Models consistently underperformed in culturally grounded areas, with Humanities and Language achieving average scores of 66.15% and 62.37% respectively with Claude 3.5 Sonnet. These results reflect the inherent challenges these models face when navigating cultural context and idiomatic expressions that are deeply embedded in local knowledge. Similarly, performance in STEM was limited (average of 61.40% for Claude), likely due to the difficulty of handling domain-specific terminology and complex conceptual reasoning. However, Social Sciences demonstrated markedly stronger performance, achieving 77.55% accuracy. This success can be attributed to the predominantly factual and descriptive nature of subjects such as Geography, Citizenship Education, and Health, which rely on standardized terminology that is well represented in the training data of most multilingual LLMs.

### Results Across Difficulty Levels
Evaluating model scores across difficulty levels indicates that Sinhala questions at the collegiate level (hard), which typically include five answer options, pose greater challenges for all models.

### Few-Shot vs. Zero-Shot
As shown in Table 4, few-shot prompting results in limited or inconsistent improvements across top-performing open-source models. While base models such as Qwen2.5-32B show marginal gains, instruction-tuned models like LLaMA-3.1-70B-chat exhibit performance degradation, likely due to their optimization for dialogue-based tasks rather than in-context learning.

| Model | 0-shot (%) | 3-shot (%) |
| :--- | :---: | :---: |
| **QWEN2.5-72B-CHAT** | 41.18 | 41.24 |
| **QWEN2.5-72B** | 35.14 | 35.20 |
| **QWEN2.5-32B-CHAT** | 36.47 | 37.54 |
| **QWEN2.5-32B** | 29.15 | 29.68 |
| **LLAMA-3.1-70B-CHAT** | 26.37 | 25.05 |
| **LLAMA-3.1-70B** | 22.40 | 22.89 |

*Table 4: Accuracy comparison of models under 0-shot and 3-shot settings.*

### Are Negations Challenging?
Negation is frequently used in Sri Lankan school exam questions to increase question difficulty and evaluate students' capacity for reasoning. In the domain of NLP, prior work has similarly demonstrated that negation poses significant challenges to model performance (Truong et al., 2023). We utilized specific negation phrases to identify questions containing negations in Sinhala, including: නොගැළපේ (Not compatible), අසත්‍ය (False), වැරදි (Wrong), and සාවද්‍ය (False).

As shown in Table 5, most models perform less effectively on questions containing negation compared to those without, including Claude 3.5 Sonnet. GPT-4o is the only model that maintains robust performance under negation.

| Model | Negations (w / w/o) | Suboptions (w / w/o) |
| :--- | :---: | :---: |
| **CLAUDE-3-5-SONNET** | 58.36 / 67.66 | 57.45 / 68.72 |
| **GPT-4o** | 59.59 / 62.91 | 53.14 / 64.05 |
| **CLAUDE-3-HAIKU** | 42.35 / 32.65 | 28.11 / 43.60 |
| **QWEN2.5-72B-CHAT** | 25.30 / 41.82 | 34.10 / 41.91 |
| **QWEN2.5-32B-CHAT** | 24.48 / 37.12 | 26.88 / 37.47 |
| **LLAMA-3.1-70B-CHAT** | 22.45 / 26.00 | 20.73 / 26.96 |

*Table 5: Average accuracy classified by questions with (w) and without (w/o) negation expressions and suboptions.*

### Impact of Suboption Questions
Similar to negation-based questions, suboption questions consist of a main question followed by multiple subparts (e.g., labeled A, B, C, D). Formats include selecting sequences of events, matching items, or analyzing logical statements. These account for about 9.8% of the data. As shown in Table 5, all models exhibit weaker performance on suboption questions due to the deeper multi-step reasoning required.

---

## 6 Do Hard Questions Get Easier with Fewer Options?

In Sri Lankan GCE A-Level exams (hard level), multiple-choice questions typically include five answer choices, unlike the standard four used in junior/senior secondary levels. To investigate whether reducing options improves performance, we compared accuracy between the original 5-option format and a modified 4-option version (where one incorrect option was randomly removed).

| Model | 5 Options | 4 Options | $\Delta$ |
| :--- | :---: | :---: | :---: |
| **CLAUDE-3-5-SONNET** | 60.11 | 66.13 | +6.02 |
| **GPT-4o** | 56.71 | 62.17 | +5.46 |
| **CLAUDE-3-HAIKU** | 34.21 | 39.44 | +5.23 |
| **QWEN2.5-72B-CHAT** | 35.72 | 40.95 | +6.74 |
| **QWEN2.5-32B-CHAT** | 30.81 | 34.73 | +3.92 |
| **LLAMA-3.1-70B-CHAT** | 22.38 | 26.81 | +4.43 |

*Table 6: Performance comparison on 5-option vs modified 4-option settings for hard questions.*

As shown in Table 6, all models improved with fewer options. However, even with 4 choices, performance remained lower than on medium-difficulty questions, suggesting that reducing options does not fully offset the complexity of hard-level content.

| Model | 4 Options (Corr) | 5 Options (Corr) |
| :--- | :---: | :---: |
| **QWEN2.5-72B-CHAT** | 0.6835 | 0.7588 |
| **QWEN2.5-32B-CHAT** | -0.0045 | 0.0696 |
| **LLAMA-3.1-70B-CHAT** | -0.0602 | -0.5529 |

*Table 7: Correlation between model confidence and accuracy under 4-option and 5-option settings.*

Interestingly, while confidence scores tend to be higher in the 4-option setup (Table 7), correlation between confidence and accuracy remains weak or inconsistent across models.

---

## 7 Do Translated STEM Questions Match the Quality of Native Ones?

Translation is the simplest way to scale multilingual benchmarks (e.g., GlobalMMLU). However, translating domain-specific terminology accurately remains a key challenge. In Sinhala, scientific concepts are conveyed through precise, standardized vocabulary set by the national curriculum. Literal machine translations often fail to capture these nuances.

To assess quality, we randomly sampled 100 native questions from SinhalaMMLU and 100 translated STEM questions from GlobalMMLU-si (excluding math/engineering). Two Sinhala-speaking annotators from the Biology stream rated each question's naturalness on a 5-point scale, converted to a 100-point scale.

| Dataset | Score (%) |
| :--- | :---: |
| **SinhalaMMLU (Ours)** | **97.30** |
| **GlobalMMLU-si** (Singh et al., 2025) | 71.07 |

*Table 8: Naturalness ratings: GlobalMMLU-si vs. SinhalaMMLU STEM questions.*

Native Sinhala content scored significantly higher (97.30%) compared to translated content (71.07%). For instance, "plasmolysis" was translated as ප්ලාස්මොලිසිස් (*plāsmolisis*) in GlobalMMLU instead of the standard technical term ප්ලාස්මච්ඡේදනය (*plāsmacchēdanaya*). This emphasizes that direct translation is insufficient for high-quality multilingual benchmarks.

---

## 8 Cultural Knowledge Analysis

To analyze how well LLMs understand Sri Lankan culture and language nuances, we annotated a subset of 1,608 questions (22% of dataset) requiring local knowledge (e.g., drama, traditional dance, oriental music, Sri Lankan history, Sinhala literature).

| Model | Original (%) | Cultural Subset (%) | $\Delta$ |
| :--- | :---: | :---: | :---: |
| **CLAUDE-3-5-SONNET** | 67.65 | 39.43 | -28.22 |
| **GPT-4o** | 62.95 | 39.12 | -23.83 |
| **CLAUDE-3-HAIKU** | 42.14 | 28.11 | -14.03 |
| **QWEN2.5-72B-CHAT** | 41.18 | 30.03 | -11.15 |
| **QWEN2.5-32B-CHAT** | 36.17 | 28.36 | -7.81 |
| **LLAMA-3.1-70B-CHAT** | 26.37 | 26.36 | -0.01 |

*Table 9: Performance comparison on the original SinhalaMMLU vs its culturally grounded subset.*

All top-performing models suffered performance drops on culturally grounded questions. Claude 3.5 Sonnet and GPT-4o dropped by 28.22 and 23.83 percentage points, respectively. Claude 3.5 Sonnet performed better on Sinhala language and Buddhist concepts, whereas GPT-4o showed relative strength in traditional music and historical knowledge.

**Error Analysis:** GPT-4o correctly answered 241 questions that Claude 3.5 Sonnet failed, mostly from History (109) and Drama/Theatre (48). Qwen2.5-72B-Chat and LLaMA-3.1-70B-Chat answered 220 and 190 of these missed questions. This suggests that while closed models understand structure well, they frequently miss localized cultural nuances.

---

## 9 Conclusion

We introduced **SinhalaMMLU**, the first multi-task language understanding benchmark for Sinhala, structured according to the Sri Lankan educational curriculum. While closed-source models achieve higher overall accuracy, all evaluated models struggle on culturally embedded concepts and language-specific features. This highlights the need for further research in culturally aware and low-resource LLMs.

---

## 10 Limitations

1. **Text-Based Questions Only:** Excludes multimodal (image, audio, video) and open-ended essay questions.
2. **Mathematics Excluded:** Math was excluded due to its lack of MCQ format in standard national exams and existing coverage in English benchmarks.
3. **Educational Scope:** Focuses primarily on secondary and collegiate levels; excludes primary and professional examinations.
4. **Human Evaluation Constraints:** Human evaluation across all 7,000+ questions was unfeasible due to the specialized domain knowledge required. National exam pass rates serve as context instead.
5. **Domain Imbalance:** Curriculum constraints resulted in Humanities and Social Sciences having higher question counts than STEM.
6. **Prompting Limitations:** Experiments did not include Chain-of-Thought (CoT) prompting or vision-language setups.

---

## 11 Ethical Statement

The SinhalaMMLU dataset was collected from publicly available government web resources. Under Section 11 of the Sri Lankan Copyright Act No. 36 of 2003, reproduction for teaching, research, and scholarly purposes constitutes fair use. The benchmark is intended strictly for research and will be released under an open license. Crowdsourced annotators received recognition in accordance with ACL guidelines.

---

## Acknowledgments

We thank our annotators, Kavindu Akalanka Wickramasinghe and Dhanushi Nagodage, for their valuable contributions to the human evaluation of STEM subjects.

---

## References

- Adelani et al. (2025). IrokoBench: A new benchmark for African languages in the age of large language models. *NAACL*.
- Anthropic. (2024). Introducing the next generation of Claude.
- Bandarkar et al. (2024). The Belebele benchmark: a parallel reading comprehension dataset in 122 language variants. *ACL*.
- Etxaniz et al. (2024). BertaQA: How much do language models know about local culture? *NeurIPS*.
- Gao et al. (2024). The Language Model Evaluation Harness.
- Ghahroodi et al. (2024). Khayyam Challenge (PersianMMLU): Is your LLM truly wise to the Persian language? *arXiv:2404.06644*.
- Hasan et al. (2021). XL-Sum: Large-scale multilingual abstractive summarization for 44 languages. *ACL-IJCNLP*.
- Hendrycks et al. (2021). Measuring massive multitask language understanding. *ICLR*.
- Hettiarachchi et al. (2024). NSina: A news corpus for Sinhala. *LREC-COLING*.
- Hewapathirana et al. (2024). M2DS: Multilingual dataset for multi-document summarisation.
- Ji et al. (2023). Cultural and Linguistic Bias of Neural Machine Translation Technology. *Cambridge University Press*.
- Jiang et al. (2024). Mixtral of experts. *arXiv:2401.04088*.
- Koto et al. (2023). Large language models only pass primary school exams in Indonesia: A comprehensive test on IndoMMLU. *EMNLP*.
- Koto et al. (2024). ArabicMMLU: Assessing massive multitask language understanding in Arabic. *ACL*.
- Li et al. (2024). CMMLU: Measuring massive multitask language understanding in Chinese. *ACL*.
- Liang et al. (2020). XGLUE: A new benchmark dataset for cross-lingual pre-training, understanding and generation. *EMNLP*.
- Liu et al. (2024). OmgEval: An open multilingual generative evaluation benchmark for large language models.
- OpenAI et al. (2024). GPT-4 technical report. *arXiv:2303.08774*.
- Poh et al. (2024). MalayMMLU: A multitask benchmark for the low-resource Malay language. *EMNLP*.
- Qwen et al. (2025). Qwen2.5 technical report. *arXiv:2412.15115*.
- Rajpurkar et al. (2016). SQuAD: 100,000+ questions for machine comprehension of text. *EMNLP*.
- Ranasinghe et al. (2024). SOLD: Sinhala offensive language dataset. *Language Resources and Evaluation*.
- Romanou et al. (2024). INCLUDE: Evaluating multilingual language understanding with regional knowledge. *arXiv:2411.19799*.
- Ruder et al. (2021). XTREME-R: Towards more challenging and nuanced multilingual evaluation. *EMNLP*.
- Sakai et al. (2024). mCSQA: Multilingual commonsense reasoning dataset with unified creation strategy. *ACL*.
- Singh et al. (2024). Aya Dataset: An open-access collection for multilingual instruction tuning. *ACL*.
- Singh et al. (2025). Global MMLU: Understanding and addressing cultural and linguistic biases in multilingual evaluation. *ACL*.
- Son et al. (2025). KMMLU: Measuring massive multitask language understanding in Korean. *NAACL*.
- Touvron et al. (2023). LLaMA: Open and efficient foundation language models. *arXiv:2302.13971*.
- Truong et al. (2023). Language models are not naysayers: an analysis of language models on negation benchmarks. **SEM*.
- Verma et al. (2025). MILU: A multi-task Indic language understanding benchmark. *NAACL*.
- Wang et al. (2018). GLUE: A multi-task benchmark and analysis platform for natural language understanding. *EMNLP Workshop*.
- Wang et al. (2019). SuperGLUE: A stickier benchmark for general-purpose language understanding systems. *NeurIPS*.
- Warusawithana et al. (2022). A systematic approach to derive a refined speech corpus for Sinhala. *LREC*.
- Winata et al. (2025). WorldCuisines: A massive-scale benchmark for multilingual and multicultural visual question answering on global cuisines. *NAACL*.
- Yüksel et al. (2024). TurkishMMLU: Measuring massive multitask language understanding in Turkish. *EMNLP*.

---

## Appendix A: Detailed Data Breakdown

| No. | Subject | Easy (Grades 6-8) | Medium (Grades 9-11) | Hard (Grades 12-13) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | History | 154 | 121 | - |
| 2 | Drama and Theatre | 131 | 140 | 143 |
| 3 | Dancing | 111 | 171 | - |
| 4 | Eastern Music | 146 | 113 | - |
| 5 | Arts | 103 | 133 | - |
| 6 | Buddhism | 165 | 120 | 147 |
| 7 | Catholicism | 135 | 119 | - |
| 8 | Christianity | 133 | 118 | 150 |
| 9 | Islam | 111 | 120 | 119 |
| 10 | Citizenship Education | 126 | 156 | - |
| 11 | Health and Physical Science | 133 | 105 | - |
| 12 | Geography | 120 | 157 | 127 |
| 13 | Science | 167 | 112 | - |
| 14 | Sinhala Language and Literature | 158 | 120 | 119 |
| 15 | Business and Accounting Studies | - | 122 | 109 |
| 16 | Entrepreneurship Studies | - | 117 | - |
| 17 | Home Economics | - | 150 | 140 |
| 18 | Communication and Media Studies | - | 119 | 112 |
| 19 | Agriculture and Food Technology | - | 136 | 158 |
| 20 | Design and Construction Technology | - | 117 | - |
| 21 | Economics | - | - | 129 |
| 22 | Biosystems Technology | - | - | 106 |
| 23 | Buddhist Civilization | - | - | 119 |
| 24 | Political Science | - | - | 160 |
| 25 | Physics | - | - | 107 |
| 26 | Chemistry | - | - | 116 |
| 27 | Biology | - | - | 127 |
| 28 | Oriental Music | - | - | 117 |
| 29 | History of Sri Lanka | - | - | 147 |
| 30 | Dancing Indigenous | - | - | 132 |

*Table 10: Subject-wise counts of questions by each difficulty level in the SinhalaMMLU dataset.*

| Educational Level | Grades | Age Range | Description |
| :--- | :---: | :---: | :--- |
| **Primary (Excluded)** | 1–5 | 5–9 | Foundational learning; excluded due to unsuitability for MCQs. |
| **Junior Secondary** | 6–9 | 10–13 | Covers core subjects (Science, Social Studies, Religion, Aesthetics, etc.). |
| **Senior Secondary (O-Level)** | 10–11 | 14–16 | Prepares for GCE O-Level exams with 6 main and 3 elective subjects. |
| **Collegiate (A-Level)** | 12–13 | 17–19/20 | 5 streams (Arts, Commerce, Bio, Physical Sci, Tech). Core qualification for university admission. |

*Table 11: Structure of the Sri Lankan national education curriculum relevant to SinhalaMMLU.*

---

## Appendix B: Experimental Setup & Models

All model generations were conducted on a single NVIDIA RTX A6000 Ada GPU using 4-bit quantization. GPT-4o experiments used top_p $= 0$ and temperature $= 0.5$.

| Model | Hugging Face ID / Source |
| :--- | :--- |
| GPT-4o | `gpt-4o-2024-08-06` |
| Claude 3.5 Sonnet | `claude-3-5-sonnet-20241022` |
| Claude 3 Haiku | `claude-3-haiku-20240307` |
| Aya Expanse 32B | `CohereForAI/aya-32b` |
| Aya Expanse 8B | `CohereForAI/aya-8b` |
| LLaMA-3.2-1B-Instruct | `meta-llama/Llama-3.2-1B-Instruct` |
| LLaMA-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` |
| LLaMA-3.3-70B-Instruct | `meta-llama/Llama-3.3-70B-Instruct` |
| LLaMA-3.2-1B | `meta-llama/Llama-3.2-1B` |
| LLaMA-3.2-3B | `meta-llama/Llama-3.2-3B` |
| LLaMA-3-70B-Instruct | `meta-llama/Meta-Llama-3-70B-Instruct` |
| LLaMA-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` |
| LLaMA-3.1-70B-Instruct | `meta-llama/Meta-Llama-3.1-70B-Instruct` |
| LLaMA-3.1-8B-Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| LLaMA-3-70B | `meta-llama/Meta-Llama-3-70B` |
| LLaMA-3-8B | `meta-llama/Meta-Llama-3-8B` |
| LLaMA-3.1-70B | `meta-llama/Meta-Llama-3.1-70B` |
| LLaMA-3.1-8B | `meta-llama/Meta-Llama-3.1-8B` |
| Mistral-7B-v0.3 | `mistralai/Mistral-7B-v0.3` |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` |
| Qwen2.5-32B | `Qwen/Qwen2.5-32B` |
| Qwen2.5-32B-Instruct | `Qwen/Qwen2.5-32B-Instruct` |
| Qwen2.5-72B | `Qwen/Qwen2.5-72B` |
| Qwen2.5-72B-Instruct | `Qwen/Qwen2.5-72B-Instruct` |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |

*Table 13: Models used in the study.*

### Prompts

**Prompt with subject information:**
```text
පහත දැක්වෙන්නේ [SUBJECT] සම්බන්ධ බහුවරණ ප්‍රශ්නයකි.
පහත දැක්වෙන ප්‍රශ්නය සඳහා නිවැරදි හෝ වඩාත්ම ගැළපෙන පිළිතුර 1, 2, 3, 4, 5 පිළිතුරුවලින් තෝරන්න.

ප්‍රශ්නය: [QUESTION]
[OPTIONS]
පිළිතුර:
```

**Prompt without subject information:**
```text
පහත දැක්වෙන්නේ බහුවරණ ප්‍රශ්නයකි.
පහත දැක්වෙන ප්‍රශ්නය සඳහා නිවැරදි හෝ වඩාත්ම ගැළපෙන පිළිතුර 1, 2, 3, 4, 5 පිළිතුරුවලින් තෝරන්න.

ප්‍රශ්නය: [QUESTION]
[OPTIONS]
පිළිතුර:
```

---

## Appendix C: Supplementary Results

### Few-Shot Accuracy Breakdown

| Model | Humanities | Language | Social Sci. | STEM | Business | Other | Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LLAMA-3-8B | 24.84 | 26.55 | 30.19 | 23.55 | 20.65 | 21.33 | 24.92 |
| LLAMA-3-8B-CHAT | 22.45 | 19.59 | 25.44 | 19.22 | 19.78 | 20.77 | 22.03 |
| LLAMA-3-70B | 22.61 | 21.39 | 25.40 | 22.16 | 20.65 | 21.77 | 22.59 |
| LLAMA-3-70B-CHAT | 23.30 | 23.16 | 24.05 | 27.20 | 19.14 | 23.51 | 23.35 |
| LLAMA-3.1-8B | 28.28 | 27.32 | 34.43 | 27.15 | 26.88 | 25.19 | 28.60 |
| LLAMA-3.1-8B-CHAT | 23.47 | 23.71 | 27.90 | 23.29 | 21.94 | 23.20 | 23.99 |
| LLAMA-3.1-70B | 23.29 | 22.68 | 25.19 | 21.16 | 19.78 | 21.44 | 22.90 |
| LLAMA-3.1-70B-CHAT | 25.16 | 26.47 | 29.15 | 23.43 | 20.86 | 23.20 | 25.05 |
| LLAMA-3.2-1B | 23.29 | 20.10 | 25.09 | 22.16 | 23.87 | 22.98 | 23.30 |
| LLAMA-3.2-1B-CHAT | 22.18 | 21.65 | 25.64 | 18.40 | 19.78 | 22.21 | 22.17 |
| LLAMA-3.2-3B | 23.86 | 20.88 | 25.00 | 21.76 | 22.80 | 22.65 | 23.47 |
| LLAMA-3.2-3B-CHAT | 22.69 | 21.39 | 25.05 | 21.17 | 20.22 | 19.89 | 22.28 |
| LLAMA-3.3-70B-CHAT | 26.09 | 24.63 | 30.31 | 24.18 | 20.00 | 21.74 | 25.45 |
| MISTRAL-7B-CHAT | 22.87 | 22.68 | 25.83 | 21.56 | 24.09 | 21.66 | 23.13 |
| QWEN2.5-32B | 28.94 | 28.87 | 32.71 | 30.62 | 25.16 | 31.05 | 29.68 |
| QWEN2.5-32B-CHAT | 37.29 | 35.05 | 41.45 | 39.74 | 32.47 | 36.24 | 37.54 |
| QWEN2.5-72B | 35.05 | 27.58 | 43.87 | 36.16 | 29.89 | 30.94 | 35.20 |
| QWEN2.5-72B-CHAT | 40.59 | 36.86 | 46.70 | 44.30 | 39.57 | 37.90 | 41.24 |
| QWEN2.5-7B | 22.18 | 21.65 | 25.64 | 18.40 | 19.78 | 22.10 | 22.15 |
| QWEN2.5-7B-CHAT | 28.23 | 26.55 | 31.04 | 26.06 | 25.38 | 28.40 | 28.18 |
| AYA-EXPANSE-32B | 24.91 | 25.74 | 30.26 | 27.20 | 21.72 | 22.40 | 25.21 |
| AYA-EXPANSE-8B | 22.30 | 21.65 | 25.64 | 17.43 | 20.22 | 21.77 | 22.11 |

*Table 14: Few-shot (3-shot) accuracy per domain.*

### Zero-Shot vs 3-Shot Comparison by Difficulty Level

| Model | Easy (0-s / 3-s) | Medium (0-s / 3-s) | Hard (0-s / 3-s) |
| :--- | :---: | :---: | :---: |
| LLAMA-3.2-1B | 25.55 / 25.77 | 23.34 / 24.74 | 18.38 / 19.76 |
| LLAMA-3.2-1B-CHAT | 25.72 / 25.54 | 23.34 / 23.38 | 18.46 / 18.42 |
| LLAMA-3.2-3B | 25.82 / 26.63 | 23.26 / 24.50 | 18.34 / 19.80 |
| LLAMA-3.2-3B-CHAT | 25.66 / 26.76 | 23.26 / 22.98 | 18.46 / 18.21 |
| LLAMA-3.3-70B-CHAT | 28.25 / 29.63 | 25.14 / 25.62 | 20.95 / 20.64 |
| LLAMA-3-70B | 26.31 / 27.11 | 24.98 / 24.06 | 18.53 / 18.41 |
| LLAMA-3-70B-CHAT | 29.07 / 25.26 | 26.50 / 25.36 | 21.11 / 19.47 |
| LLAMA-3-8B | 26.26 / 29.66 | 23.78 / 26.50 | 18.50 / 19.41 |
| LLAMA-3-8B-CHAT | 24.38 / 27.17 | 23.46 / 23.82 | 19.01 / 18.79 |
| LLAMA-3.1-70B | 26.09 / 25.72 | 23.46 / 24.66 | 18.46 / 18.71 |
| LLAMA-3.1-70B-CHAT | 30.42 / 28.80 | 26.14 / 27.41 | 22.38 / 19.81 |
| LLAMA-3.1-8B | 34.31 / 26.26 | 24.98 / 30.65 | 19.72 / 21.80 |
| LLAMA-3.1-8B-CHAT | 29.44 / 28.08 | 25.14 / 26.78 | 20.75 / 19.74 |
| MISTRAL-7B-CHAT | 24.49 / 25.15 | 24.46 / 25.46 | 18.37 / 19.02 |
| QWEN2.5-32B | 34.58 / 36.26 | 30.81 / 27.77 | 23.52 / 26.74 |
| QWEN2.5-32B-CHAT | 42.30 / 41.57 | 40.42 / 37.87 | 30.81 / 31.54 |
| QWEN2.5-72B | 41.06 / 49.05 | 25.54 / 37.55 | 28.40 / 34.60 |
| QWEN2.5-72B-CHAT | 45.11 / 46.25 | 43.85 / 43.77 | 35.72 / 34.69 |
| QWEN2.5-7B | 25.66 / 25.54 | 23.30 / 23.42 | 18.46 / 18.42 |
| QWEN2.5-7B-CHAT | 30.85 / 31.01 | 28.25 / 30.41 | 23.56 / 23.76 |
| AYA-EXPANSE-32B | 25.28 / 28.75 | 27.89 / 25.54 | 22.30 / 19.60 |
| AYA-EXPANSE-8B | 24.85 / 25.59 | 24.14 / 23.50 | 19.49 / 18.05 |

*Table 15: Model Performance Comparison (Zero-Shot vs 3-Shot across difficulty levels).*