🏥 Advanced Medical Imaging Diagnosis Agent

**🔍 System Overview**

The Medical Imaging System is an AI-powered Streamlit web application designed to assist healthcare professionals with medical image analysis. This comprehensive platform analyzes diverse medical images, provides AI-generated diagnoses with explanatory visualizations, connects to medical knowledge bases, and enables multi-doctor collaboration.

**Demo Link**
https://drive.google.com/file/d/1pRvlsBIKweLYzcU8TKunQLnUwcfXSg9L/view?usp=sharing

**🧩 Main Components**

📸Image Processing:Handles multiple formats (DICOM, NIfTI, JPG/PNG)

🤖AI Analysis: Leverages OpenAI API for intelligent image interpretation

🔆XAI Visualization:Creates heatmaps highlighting influential diagnostic regions

📚Knowledge Integration:Connects to PubMed and clinical trials databases

👨👩⚕️Multi-doctor Collaboration:Real-time chat system for case discussions

📄Report Generation:Creates downloadable PDF reports with findings

🗃️Archive & Search:Stores and retrieves historical analyses


**⚙️ Technology Stack**

🖥️Frontend:Streamlit for intuitive web interface

🔧Image Processing:PyDICOM, Nibabel, OpenCV, PIL

🧠AI Integration:OpenAI API for analysis

📊Data Handling:Pandas, NumPy for data manipulation

📈Visualization:Matplotlib, custom heatmap generation

📝Report Generation:ReportLab for PDF creation

💾Storage:JSON-based file storage system

🔍Knowledge Base:PubMed API (via Biopython)



**User Journey**

⚙️ Setup: Configure API keys and enable desired features.

📤 Upload: Select and upload medical image files.

🔍 Analyze: Trigger AI analysis of the image.

📊 Review: Examine findings, heatmaps, and literature.

👥 Collaborate: Discuss with colleagues through the chat system.

📋 Document: Generate reports and save to the archive.

**Benefits for Medical Professionals**

🩺 Diagnostic Support: AI-assisted analysis provides a second opinion.

⏱️ Efficiency: Automates literature search and report generation.

🔍 Explainability: Heatmaps show how the AI reached conclusions.

👥 Collaboration: Multiple specialists can discuss complex cases.

📚 Knowledge Access: Integration with medical literature databases.

📝 Documentation: Structured reports for medical records.


**Implementation Details**

🗄️ Storage System:

analysis_store.json — archives all case analyses with metadata.

chat_store.json — maintains collaboration rooms and messages.

🔐 Configuration:

OpenAI API key setup in the sidebar.

Toggle-based enablement for XAI, reports, and knowledge base.

🔄 Mock Services (development):

Simulated PubMed/clinical-trials responses.

AI-generated analyses.

UI Organization

📑 Tabs-based interface:

Image Analysis — uploads and results.

Collaboration — multi-doctor chat.

Search Archives — historical case lookup.

📊 Analysis Display:

Split view with image preview and results.

Expandable sections for findings and literature.

XAI visualization with multiple viewing options.



**=>System Architecture and Data Flow**

<img width="840" height="704" alt="image" src="https://github.com/user-attachments/assets/06279dda-9b64-420d-b5e8-e14ef22b9fa3" />

<img width="854" height="646" alt="image" src="https://github.com/user-attachments/assets/44b31322-286f-4684-bec7-f7c52b174301" />

<img width="836" height="675" alt="image" src="https://github.com/user-attachments/assets/575ab5b3-54a0-4e94-8108-a2eaab1c4e65" />

<img width="916" height="743" alt="image" src="https://github.com/user-attachments/assets/0014ebd5-1ee2-4c53-8571-0cfe65eec3ab" />

