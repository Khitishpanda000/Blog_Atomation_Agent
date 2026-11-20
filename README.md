# Automated AI Blog Generation Agent (LangGraph)

An end-to-end agentic workflow built with **LangGraph** that automates the entire blog creation lifecycle—from initial topic planning and live web research to structured Markdown writing and image integration.

## 🚀 Features

* **Structured Planning:** Breaks down a given topic into a logical outline and section-by-section writing plan.
* **Live Web Research:** Integrates **Tavily Search** to gather up-to-date facts, context, and references before drafting.
* **Iterative Pipeline Evolution:** Documented step-by-step across four progressive Jupyter Notebooks, moving from basic generation to prompt optimization, research capabilities, and multimodal/image support.
* **Markdown Output:** Automatically compiles and exports clean, publish-ready `.md` blog posts.

## 📂 Project Structure

```text
├── 1_Blog_Atomation_Agent_Basics.ipynb             # Core LangGraph workflow for basic outline & drafting
├── 2_Blog_Atomation_Agent_Improved_Prompts.ipynb   # Enhanced prompt engineering for better tone & structure
├── 3_Blog_Atomation_Agent_research.ipynb           # Integration of Tavily web search for factual grounding
├── 4_Blog_Atomation_Agent_Image.ipynb              # Final pipeline with automated image sourcing/generation
├── tavily_test.ipynb                               # Sandbox notebook for testing Tavily search queries
├── Self_Attention_Blog.md                          # Sample generated output (v1)
├── Self_AttentionX.md                              # Sample generated output (v2 - improved prompts)
├── State of Multimodal LLMs in 2026.md             # Sample generated output with web research & media
└── requirements.txt                                # Project dependencies
```

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd your-repo-name
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .myenv
   # On Windows:
   .myenv\Scripts\activate
   # On macOS/Linux:
   source .myenv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
   *(Note: Update the variable names above if you used a different LLM or image provider such as Gemini, Anthropic, or Stability AI).*

## 💻 Usage

1. Open any of the numbered notebooks (start with `4_Blog_Atomation_Agent_Image.ipynb` for the full feature set) in Jupyter Notebook or VS Code.
2. Select your `.myenv` kernel.
3. Input your desired blog topic in the execution cell and run all cells to generate a complete Markdown file in the project directory.

## 🔒 Note Before Pushing to Git

Make sure to add a `.gitignore` file to your repository so you don't accidentally upload your virtual environment or secret API keys:
```text
.myenv/
__pycache__/
.env
.ipynb_checkpoints/
```
