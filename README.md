# AI Remote Troubleshooting System

An AI-powered troubleshooting platform that converts voice-based issue descriptions into actionable technical guidance using speech-to-text and Large Language Models (LLMs). The system streamlines support workflows by automating transcription, issue summarization, and troubleshooting recommendations through an interactive Streamlit interface.

## 🚀 Features

- 🎤 Voice-based issue reporting
- 📝 Automatic speech-to-text transcription using Whisper AI
- 🤖 AI-generated troubleshooting recommendations
- 📄 Automated issue summarization
- ⚡ Real-time support workflow management
- 📊 Interactive Streamlit dashboard
- 🔍 Faster issue diagnosis and resolution

## 🛠️ Tech Stack

- Python
- Streamlit
- Whisper AI
- Groq API
- Large Language Models (LLMs)

## 📂 Project Structure

```text
AI-Remote-Troubleshooting-System/
│
├── app.py
├── requirements.txt
├── assets/
├── modules/
├── utils/
├── data/
└── README.md
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Remote-Troubleshooting-System.git
cd AI-Remote-Troubleshooting-System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux/Mac**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will start locally and can be accessed through your browser.

## 📈 Workflow

1. User describes a technical issue through voice input.
2. Whisper AI converts speech into text.
3. The LLM analyzes the issue description.
4. The system generates troubleshooting recommendations.
5. An issue summary is created automatically.
6. Results are displayed through the Streamlit dashboard.

## 🎯 Use Cases

- IT Help Desk Support
- Remote Technical Assistance
- Device Troubleshooting
- Customer Support Automation
- Knowledge Base Assistance

## 💡 Key Outcomes

- Reduced manual troubleshooting effort
- Improved support response efficiency
- Automated documentation and summaries
- Enhanced user experience through voice interaction

## 📸 Demo

Add screenshots or GIFs of the application here.

## 🔮 Future Enhancements

- Multi-language support
- Knowledge base integration
- Ticket management system
- Conversation history tracking
- Advanced analytics dashboard

## 👨‍💻 Author

Parikshith V M

## 📄 License

This project is licensed under the MIT License.
