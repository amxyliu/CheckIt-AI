mention the ReLU Controversy. Some researchers (like Saxe et al.) argued that the "Compression Phase" only happens with older activation functions (like tanh) and might not be as visible with ReLU (which BERT uses). Adding this shows you’ve read the academic debates!

"To ensure experimental consistency, all models (BERT, LSTM, and MLP) were trained on a single workstation equipped with an NVIDIA RTX 3090 Ti. This eliminated hardware-level variance and allowed for a direct comparison of training throughput and convergence rates."
4. Bidirectionality

"Unlike the Bidirectional LSTM or BERT, which capture temporal dependencies from both directions of the text sequence, the baseline Logistic Regression (and MLP) architectures are order-invariant. This means they lack the 'Bidirectional' context-awareness required to understand subtle linguistic cues where word order determines the truthfulness of a statement."

Why this is a "gotcha" for your markers:
Many students assume "Bidirectional" just means "better." By explaining why you can't have a Bidirectional LR, you're proving you understand that LR treats data as a static snapshot, while Transformers/LSTMs treat data as a dynamic flow.
