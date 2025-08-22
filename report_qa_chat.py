import json
import os
import uuid
from datetime import datetime
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_PATH = os.path.join(DATA_DIR, "analysis_store.json")

# QA System for Medical Reports
class ReportQASystem:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.conversation_history = []
        self.analysis_store = self.load_analysis_store()
    
    def load_analysis_store(self):
        if os.path.exists(ANALYSIS_PATH):
            with open(ANALYSIS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"analyses": []}
    
    def get_embeddings(self, text, model="text-embedding-3-small"):
        if not self.api_key:
            raise RuntimeError("OpenAI API key not provided. Cannot compute embeddings.")
        client = openai.OpenAI(api_key=self.api_key)
        resp = client.embeddings.create(input=text, model=model)
        v = np.array(resp.data[0].embedding, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def get_relevant_contexts(self, query, top_k=3):
        """Find relevant contexts for a query using embeddings similarity"""
        # Get query embedding
        self.analysis_store = self.load_analysis_store()  # refresh from disk
        analyses = self.analysis_store.get("analyses", [])
        if not analyses:
            return ["No previous analyses found."]
        query_embedding = self.get_embeddings(query)
        
        # Extract all analyses and get embeddings
        analyses = self.analysis_store["analyses"]
        contexts = []
        
        if not analyses:
            return ["No previous analyses found."]
        
        # Get embeddings for each analysis
        for analysis in analyses:
            analysis_text = analysis.get("analysis", "")
            if not analysis_text.strip():
                continue
                
            # Combine analysis with findings
            full_text = analysis_text
            if "findings" in analysis and analysis["findings"]:
                findings_text = "\n".join([f"- {finding}" for finding in analysis["findings"]])
                full_text += f"\n\nFindings:\n{findings_text}"
            
            # Add metadata
            full_text += f"\n\nImage: {analysis.get('filename', 'unknown')}"
            full_text += f"\nDate: {analysis.get('date', '')[:10]}"
            
            contexts.append({
                "text": full_text,
                "embedding": self.get_embeddings(full_text),
                "id": analysis.get("id", ""),
                "date": analysis.get("date", "")
            })
        
        # Calculate similarities
        similarities = []
        for context in contexts:
            similarity = cosine_similarity(
                [query_embedding], 
                [context["embedding"]]
            )[0][0]
            similarities.append((similarity, context))
        
        # Sort by similarity and get top_k
        similarities.sort(reverse=True)
        top_contexts = [context["text"] for _, context in similarities[:top_k]]
        
        return top_contexts
    
    def answer_question(self, question):
        """Answer a question about medical reports using RAG"""
        if not self.api_key:
            return "Please provide an OpenAI API key to enable the QA system."
        
        # Get relevant contexts
        contexts = self.get_relevant_contexts(question)
        
        if not contexts or contexts[0] == "No previous analyses found.":
            return "I don't have any medical reports to reference. Please upload and analyze some images first."
        
        # Create combined context text
        combined_context = "\n\n---\n\n".join(contexts)
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": question})
        
        try:
            # Create prompt for GPT
            client = openai.OpenAI(api_key=self.api_key)
            
            system_prompt = f"""You are a medical AI assistant answering questions about medical reports.
            Use the following medical report contexts to answer the question.
            If the answer cannot be found in the contexts, say so and suggest what other information might be needed.
            
            Contexts:
            {combined_context}
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                *self.conversation_history
            ]
            
            # Get response from OpenAI
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 10:
                # Keep only the 10 most recent messages
                self.conversation_history = self.conversation_history[-10:]
            
            return answer
        
        except Exception as e:
            return f"I encountered an error while answering your question: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared."


# Chat room system specifically for QA
class ReportQAChat:
    def __init__(self):
        self.qa_chat_store = self.get_qa_chat_store()
    
    def get_qa_chat_store(self):
        """Get the QA chat storage"""
        if os.path.exists("qa_chat_store.json"):
            with open("qa_chat_store.json", "r") as f:
                return json.load(f)
        return {"rooms": {}}
    
    def save_qa_chat_store(self):
        """Save the QA chat storage"""
        with open("qa_chat_store.json", "w") as f:
            json.dump(self.qa_chat_store, f)
    
    def create_qa_room(self, user_name, room_name):
        """Create a new QA chat room"""
        room_id = f"QA-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        room_data = {
            "id": room_id,
            "name": room_name,
            "created_at": datetime.now().isoformat(),
            "creator": user_name,
            "messages": []
        }
        
        # Add welcome message
        welcome_message = {
            "id": str(uuid.uuid4()),
            "user": "Report QA System",
            "content": f"Welcome to the Report QA room: {room_name}. You can ask questions about your medical reports and I'll try to answer based on the analyses stored in the system.",
            "timestamp": datetime.now().isoformat()
        }
        room_data["messages"].append(welcome_message)
        
        # Store room
        self.qa_chat_store["rooms"][room_id] = room_data
        self.save_qa_chat_store()
        
        return room_id
    
    def add_message(self, room_id, user_name, message):
        """Add a message to a QA room"""
        if room_id not in self.qa_chat_store["rooms"]:
            return None
        
        message_data = {
            "id": str(uuid.uuid4()),
            "user": user_name,
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        
        self.qa_chat_store["rooms"][room_id]["messages"].append(message_data)
        self.save_qa_chat_store()
        
        return message_data
    
    def get_messages(self, room_id, limit=50):
        """Get the most recent messages from a QA room"""
        if room_id not in self.qa_chat_store["rooms"]:
            return []
        
        messages = self.qa_chat_store["rooms"][room_id]["messages"]
        return messages[-limit:] if len(messages) > limit else messages
    
    def get_qa_rooms(self):
        """Get all QA rooms"""
        rooms = []
        
        for room_id, room_data in self.qa_chat_store["rooms"].items():
            rooms.append({
                "id": room_id,
                "name": room_data.get("name", "Unnamed Room"),
                "creator": room_data.get("creator", "Unknown"),
                "created_at": room_data.get("created_at", "")
            })
        
        # Sort by creation date (newest first)
        rooms.sort(key=lambda x: x["created_at"], reverse=True)
        return rooms
    
    def delete_qa_room(self, room_id):
        """Delete a QA chat room"""
        if room_id in self.qa_chat_store["rooms"]:
            del self.qa_chat_store["rooms"][room_id]
            self.save_qa_chat_store()
            return True
        return False