**System Architecture / Workflow Diagram**
+----------------------------------------------------------------------------------+
|                                 MENTORBOT (Dynamic)                              |
+----------------------------------------------------------------------------------+
|                                                                                  |
| 1 User Input → Learner's question or response                                   |
|                                                                                  |
| 2️ Understanding Analyzer                                                        |
|    ├─ NLP models compute:                                                        |
|    │   • Vocabulary complexity (textstat, POS depth)                             |
|    │   • Domain classification (math, science, etc.)                             |
|    │   • Question reasoning type (factual vs conceptual)                         |
|    ├─ Output: Learner Level ("Beginner", "Intermediate", "Advanced")             |
|                                                                                  |
| 3️ Adaptive Profile Manager (JSON Memory)                                       |
|    ├─ Maintains user's current difficulty, accuracy, and preferences             |
|    ├─ Updates trend after each interaction                                       |
|                                                                                  |
| 4️ Context Builder (LangChain / LlamaIndex)                                     |
|    ├─ Retrieves relevant examples from Knowledge Base                            |
|    ├─ Builds adaptive prompt:                                                    |
|    │     "Explain this like you're teaching a {level} student."                  |
|                                                                                  |
| 5️ Base LLM (Hugging Face Model)                                                |
|    ├─ Model: Mistral / Phi-3 / Llama-3                                           |
|    ├─ Generates personalized explanation / feedback                              |
|                                                                                  |
| 6️ Evaluation & Feedback Loop                                                   |
|    ├─ Evaluates correctness, clarity, and engagement                             |
|    ├─ Updates learner profile and level dynamically                              |
|                                                                                  |
+----------------------------------------------------------------------------------+


**MentorBot Project Folder Structure**

MentorBot/
│
├── data/
│   ├── datasets/
│   │   ├── deepmind_math/               # Sampled data for evaluation
│   │   └── openbookqa/                  # Science reasoning dataset
│   │
│   ├── knowledge_base/                  # Custom study material or text corpus
│   └── user_profiles/                   # JSON memory per learner
│       ├── user_1.json
│       └── user_2.json
│
├── models/
│   ├── base_llm/                        # Hugging Face model weights (auto-downloaded)
│   ├── analyzer/                        # Text classification or understanding models
│   └── embeddings/                      # Vector storage (FAISS / Chroma)
│
├── src/
│   ├── main.py                          # Entry point: runs the full MentorBot pipeline
│   │
│   ├── llm/                             # Core model logic
│   │   ├── __init__.py
│   │   ├── model_loader.py              # Loads HF model (Mistral, Phi-3, etc.)
│   │   ├── adaptive_prompt_builder.py   # Builds prompts dynamically
│   │   ├── prompt_templates.py          # Stores reusable prompt formats
│   │   └── response_parser.py           # Cleans/validates model outputs
│   │
│   ├── memory/                          # Learner profile & context memory
│   │   ├── __init__.py
│   │   ├── user_profile_manager.py      # Load, update, save learner JSON profile
│   │   ├── context_retriever.py         # Retrieves relevant content using LangChain/LlamaIndex
│   │   └── memory_store.py              # Optional: cache or multi-user memory
│   │
│   ├── analysis/                        # Learner understanding and domain detection
│   │   ├── __init__.py
│   │   ├── level_detector.py            # Detects learner level (beginner/intermediate/advanced)
│   │   ├── text_complexity.py           # Uses textstat, POS depth, Flesch-Kincaid
│   │   └── domain_classifier.py         # (Optional) Detects subject area (math, science)
│   │
│   ├── evaluation/                      # Evaluate learning performance and improvement
│   │   ├── __init__.py
│   │   ├── evaluator.py                 # Accuracy, clarity, engagement metrics
│   │   ├── feedback_generator.py        # Personalized feedback text
│   │   ├── progression_tracker.py       # Learner progress updates
│   │   └── stats_tests.py               # Statistical tests (paired t-test, etc.)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                    # Paths, model names, constants
│   │   ├── logger.py                    # Logging and interaction tracking
│   │   └── helpers.py                   # Helper functions (formatting, metrics)
│   │
│   └── pipeline.py                      # Orchestrates Analyzer → Context → LLM → Feedback
│
├── ui/
│   ├── app.py                           # Streamlit or Gradio interface
│   └── components/                      # Chat UI, progress graph, etc.
│
├── experiments/
│   ├── experiment_runner.py             # Run baseline vs personalized setups
│   ├── baseline_results.json
│   ├── personalized_results.json
│   └── ablation_study.py                # Compare models / retrievers
│
├── reports/
│   ├── architecture_diagram.png
│   ├── workflow.png
│   ├── evaluation_report.pdf
│   └── slides/
│       └── mentorbot_presentation.pptx
│
├── tests/
│   ├── test_analyzer.py
│   ├── test_prompt_builder.py
│   ├── test_profile_manager.py
│   ├── test_feedback.py
│   └── test_end_to_end.py
│
├── scripts/
│   ├── download_datasets.py
│   ├── preprocess_data.py
│   └── run_pipeline.sh
│
├── requirements.txt
├── README.md
└── .gitignore


### Base LLM (Tutor Brain)
* mistralai/Mistral-7B-Instruct-v0.3
* microsoft/Phi-3-mini-4k-instruct
* meta-llama/Meta-Llama-3-8B-Instruct
### User Understanding / Level Detection
* facebook/bart-large-mnli
* google/flan-t5-base
### Domain / Subject Classifier (Optional)
* google/flan-t5-base
### Text Embeddings for Context Retrieval
* sentence-transformers/all-MiniLM-L6-v2
* intfloat/e5-base-v2
### Feedback / Evaluation Generator (Optional)
* Reuse Base LLM (Mistral-7B, Phi-3-mini, Llama-3-8B-Instruct)
