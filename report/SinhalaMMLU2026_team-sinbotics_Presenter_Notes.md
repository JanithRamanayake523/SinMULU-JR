# SinBotics presenter script

Time limit: 8 minutes, then 2 minutes of Q&A. Target pace: about 7 minutes 15 seconds.

## Slide 1: Title (15 seconds)

Hello everyone! We are Team SinBotics. Throughout this shared task, one main question guided our work: can thinking in English improve answers to Sinhala questions? Let's find out!

## Slide 2: Approach (33 seconds)

You all know this challenge well, so I will go straight to our final system. After that, I will share what we learned from our experiments.

Our final system has three main parts. First, we keep the original Sinhala, because translation can lose meaning. Second, the model reasons in English, like a Sri Lankan teacher. Third, we add exam knowledge with a small adapter.

The result? A Macro Avg of 57.58, with just 4.24 billion parameters.

## Slide 3: Models and data (25 seconds)

Our whole system is one model: Qwen3.5-4B plus a rank-16 LoRA adapter. That is only about half of the limit!

We trained on 25,799 questions from two sources: translated Global-MMLU questions, and our own real exam collection. And to be clear: no Sinhala MMLU data was used for training.

## Slide 4: Sinmulu_si (25 seconds)

That collection is Sinmulu_si, and we are proud of it. We started with 3,486 exam papers. Claude Sonnet read each page, we selected the multiple-choice questions, cleaned them, and removed overlap with the test set. That gave us 15,190 real exam questions.

And look at the effect: plus 6.2 points from real exam questions, against just 1.4 from translated ones. Real questions teach more!

## Slide 5: Training (30 seconds)

Now, how did we train? We fine-tuned a LoRA adapter for one epoch: 807 steps and 8.8 hours on one RTX 3090.

Here is the key decision: each example is a plain multiple-choice prompt, and the loss is on the answer digit alone. Nothing else.

And it worked. Watch the chart: validation accuracy climbs from 44.4 to 54.8. After training, we merged the adapter into the model.

## Slide 6: Inference (30 seconds)

At inference, four steps. We clean the text, but we never translate it. We build the teacher prompt. The model thinks in English for up to 4,096 tokens. Then we read the most probable digit, from 1 to 4. So every question gets an answer, even if the reasoning is cut off.

One pitfall worth sharing: a single trailing space took our label mass from 0.05 to 1.000. One space! Always check your scoring.

## Slide 7: Final prompt (20 seconds)

And this is the exact prompt. The first line makes the model an expert Sri Lankan teacher. The instructions are in English, and they ask it to think step by step. The question and options, in orange, stay in Sinhala. No system prompt, no few-shot examples. Simple, and powerful.

## Slide 8: Results (25 seconds)

So how did it do? Macro Avg 57.58: Easy 54.67, Medium 52.30, and Hard 65.77, which placed us 11th.

Hard was our strongest level, which is a surprise, and we did not investigate why. Our own 300-question screening set shows a very similar pattern. I will explain that set shortly.

## Slide 9: Four phases (15 seconds)

Now the most exciting part: how we got here! We tested about 70 configurations in four phases, and each phase answered one clear question. Small models, a study without training, fine-tuning, and finally a stronger base model.

## Slide 10: Evaluation datasets (30 seconds)

We used three evaluation datasets. Phase 1 used the 1,851 public Easy questions. Phase 2 split those into 280 questions for screening and 1,483 held out for confirmation. The shared-task data has no labelled Medium or Hard questions, so for phases 3 and 4 we built screening set 2: 300 Sinmulu_si questions, 100 from each grade band, removed before training.

And phase 1? All three small models were at chance. So we moved up to 4B.

## Slide 11: Conditions (12 seconds)

In phase 2 we asked: what can a 4B model gain with no training at all? We tested 59 conditions, across prompts, scoring, reasoning, translation and ensembles.

## Slide 12: Screening set 1 (20 seconds)

And here is the first big signal! The baseline scored 37.5, and the top four conditions all reason in English, reaching 50.7. Calibration and cloze fell below the baseline.

But this set is small, and some results can look good by luck. So we did not stop here.

## Slide 13: Confirmation (25 seconds)

We tested the best conditions on 1,483 held-out questions. English reasoning with a teacher role went from 38.2 to 45.6: a confirmed gain of 7.4 points!

And confirmation changed the story. Worked examples fell from 50.7 to 44.8. Without this step, we would have believed a false win.

## Slide 14: Full systems (25 seconds)

Now full systems. Moving from Gemma to Qwen gave plus 7.4 points. Thinking mode added 4.7, though that is not significant. And the surprise: the adapter added only 0.3 on reasoning prompts.

Overall, the final system beat our Submission 2 by 12.4 points, with p of 0.001. That is a big jump!

## Slide 15: Analysis (20 seconds)

Here is the whole picture. Two changes gave clear gains: English reasoning and a stronger base model. The adapters were clear only on validation. And most other methods gave no gain. Negative results matter too, and we report them all.

## Slide 16: Subjects (20 seconds)

Accuracy varies widely by subject. Our final system reaches 76 percent on social science and 73 on STEM, but 44 on Sinhala language. English reasoning helps content subjects, but it loses something when the question is about Sinhala itself.

## Slide 17: Errors and limitations (25 seconds)

So where do we fail? When reasoning finishes, accuracy is 77.5 percent. When it is cut off, it falls to 44.4, and that happened on 54 percent of questions. So a higher token limit may give better accuracy.

We also know our limits: a small selection set, partly model-derived labels, and official development data that is Easy only.

## Slide 18: Conclusion (25 seconds)

So, can thinking in English improve answers to Sinhala questions? Yes! English reasoning gave a confirmed gain of 7.4 points, without any training. Real exam data beats translated data, and held-out confirmation protects you from false wins.

Next, we want a larger Medium and Hard development set, a study of the token limit, and adapters trained on short English rationales.

## Slide 19: Thank you (5 seconds)

Thank you! Our code, predictions and adapter are public, and we are happy to take your questions.
