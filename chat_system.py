import streamlit as st
from datetime import datetime
import json
import os
import uuid
import time
import openai
from dotenv import load_dotenv

# -----------------------------
# Boot
# -----------------------------
load_dotenv()

# -----------------------------
# Simple JSON "DB"
# -----------------------------
def get_chat_store():
    """Get the chat storage from disk (or initialize)."""
    if os.path.exists("chat_store.json"):
        with open("chat_store.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {"rooms": {}}

def save_chat_store(store):
    """Persist the chat storage."""
    with open("chat_store.json", "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)

def get_messages(case_id):
    """Retrieve messages from a specific chat room."""
    store = get_chat_store()
    if case_id in store["rooms"]:
        return store["rooms"][case_id]["messages"]
    return []

def create_chat_room(case_id, creator_name, case_description):
    """Create a new chat room for a case (idempotent)."""
    store = get_chat_store()
    if case_id not in store["rooms"]:
        room_data = {
            "id": case_id,
            "created_at": datetime.now().isoformat(),
            "creator": creator_name,
            "description": case_description,
            "participants": [creator_name, "Dr. AI Assistant", "Dr. Johnson", "Dr. Chen", "Dr. Patel"],
            "messages": []
        }
        welcome_message = {
            "id": str(uuid.uuid4()),
            "user": "Dr. AI Assistant",
            "content": (
                f"Welcome to the case discussion for '{case_description}'. "
                "I've analyzed the image and I'm here to assist with the diagnosis. "
                "Feel free to ask me specific questions about the findings."
            ),
            "type": "text",
            "timestamp": datetime.now().isoformat()
        }
        room_data["messages"].append(welcome_message)
        store["rooms"][case_id] = room_data
        save_chat_store(store)
    return case_id

def join_chat_room(case_id, user_name):
    """Join an existing chat room."""
    store = get_chat_store()
    if case_id in store["rooms"]:
        if user_name not in store["rooms"][case_id]["participants"]:
            store["rooms"][case_id]["participants"].append(user_name)
            save_chat_store(store)
        return True
    return False

def add_message(case_id, user_name, message, message_type="text"):
    """Append a message to a room."""
    store = get_chat_store()
    if case_id in store["rooms"]:
        message_data = {
            "id": str(uuid.uuid4()),
            "user": user_name,
            "content": message,
            "type": message_type,
            "timestamp": datetime.now().isoformat()
        }
        store["rooms"][case_id]["messages"].append(message_data)
        save_chat_store(store)
        return message_data
    return None

def get_available_rooms():
    """List all rooms with metadata."""
    store = get_chat_store()
    rooms = []
    for room_id, room_data in store["rooms"].items():
        rooms.append({
            "id": room_id,
            "description": room_data["description"],
            "creator": room_data["creator"],
            "created_at": room_data["created_at"],
            "participants": len(room_data["participants"])
        })
    rooms.sort(key=lambda x: x["created_at"], reverse=True)
    return rooms

# -----------------------------
# OpenAI helper
# -----------------------------
def get_openai_response(user_question, case_description, findings=None, api_key=None):
    """Get a response from OpenAI based on the medical context and user question."""
    if not api_key:
        return "Please configure your OpenAI API key in the sidebar to get AI responses."

    client = openai.OpenAI(api_key=api_key)

    findings_text = ""
    if findings and len(findings) > 0:
        findings_text = "The key findings in the image are:\n"
        for i, finding in enumerate(findings, 1):
            findings_text += f"{i}. {finding}\n"

    system_prompt = f"""You are Dr. AI Assistant, a medical specialist analyzing a medical image.
The image is from a case described as: "{case_description}".
{findings_text}

Please provide an expert, accurate, and helpful response to the doctor's question.
Base your response on the findings and your medical expertise.
Respond as if you are speaking directly to the doctor in a collaborative setting.
Keep your response concise but informative, focusing on the relevant medical details.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return (
            "I apologize, but I encountered an error while analyzing your question. "
            f"Please try again or rephrase your question. Error details: {str(e)}"
        )

# -----------------------------
# UI helpers
# -----------------------------
def doctor_avatar(name: str) -> str:
    # Fun, distinct emojis per doctor
    mapping = {
        "Dr. AI Assistant": "🤖",
        "Dr. Johnson": "👩‍⚕️",   # Physician
        "Dr. Chen": "👩‍⚕️",      # Orthopaedic 
        "Dr. Patel": "👩‍⚕️",     # radiologist
    }
    return mapping.get(name, "👩‍⚕️")  # generic doctor

def render_message_bubble(msg, current_user_name: str):
    """Render a single chat bubble with the speaker's name and content."""
    role = "user" if msg["user"] == current_user_name else "assistant"
    avatar = doctor_avatar(msg["user"]) if msg["user"] != current_user_name else "🧑‍⚕️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f"**{msg['user']}**")
        if msg.get("type") == "annotation":
            st.write("📝 **Image Annotation:**")
        st.write(msg["content"])

# -----------------------------
# Main UI
# -----------------------------
def render_chat_interface():
    st.subheader("👨‍⚕️👩‍⚕️ Multi-Doctor Collaboration")

    # Current user
    if "user_name" not in st.session_state:
        st.session_state.user_name = "Dr Shah"  # your user name
    user_name = st.text_input("Your Name", value=st.session_state.user_name, key="user_name_input")
    if user_name != st.session_state.user_name:
        st.session_state.user_name = user_name

    # Persist doctor selection
    st.session_state.setdefault("selected_doctor", None)

    # Tabs
    tab1, tab2 = st.tabs(["Join Existing Case", "Create New Case"])

    with tab1:
        rooms = get_available_rooms()
        if rooms:
            room_options = {f"{room['id']} - {room['description']} (by {room['creator']})": room["id"] for room in rooms}
            selected_room_label = st.selectbox("Select Case", options=list(room_options.keys()), key="room_select")
            if st.button("Join Discussion", key="join_discussion_btn"):
                selected_case_id = room_options[selected_room_label]
                if join_chat_room(selected_case_id, user_name):
                    st.session_state.current_case_id = selected_case_id
                    st.rerun()
        else:
            st.info("No active case discussions. Create a new one!")

    with tab2:
        case_description = st.text_input("Case Description", key="new_case_desc")

        # Only allow create when an image has been uploaded + processed by your pipeline
        can_create_discussion = (
            "file_data" in st.session_state and
            "file_type" in st.session_state and
            st.session_state.file_type is not None
        )

        if can_create_discussion:
            case_id = f"{st.session_state.file_type.upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            if st.button("Create Discussion", key="create_discussion_btn"):
                if case_description.strip():
                    created_case_id = create_chat_room(case_id, user_name, case_description.strip())
                    st.session_state.current_case_id = created_case_id
                    st.rerun()
                else:
                    st.error("Please provide a case description")
        else:
            if "file_data" not in st.session_state:
                st.info("Upload an image first to create a new case discussion")
            elif "file_type" not in st.session_state or st.session_state.file_type is None:
                st.info("Please complete the image upload and processing before creating a discussion")
            else:
                st.info("Upload an image first to create a new case discussion")

    # Active chat
    if "current_case_id" in st.session_state:
        case_id = st.session_state.current_case_id
        store = get_chat_store()

        if case_id in store["rooms"]:
            room_data = store["rooms"][case_id]

            st.subheader(f"Case Discussion: {room_data['description']}")
            st.caption(f"Created by {room_data['creator']} • {len(room_data['participants'])} participants")

            # -----------------------------
            # Response options
            # -----------------------------
            response_col1, response_col2 = st.columns(2)

            with response_col1:
                # If AI response is on, we won't allow doctor selection
                get_ai_response = st.checkbox("Get AI Assistant Response", value=True, key=f"opt_ai_resp_{case_id}")

            with response_col2:
                selected_doc = None
                if get_ai_response:
                    doctor_response = False
                    st.session_state["selected_doctor"] = None
                else:
                    doctor_response = st.checkbox("Get Doctor Response", value=True, key=f"opt_doc_resp_{case_id}")
                    if doctor_response:
                        raw_choice = st.selectbox(
                            "Select Doctor",
                            ["Dr. Johnson (Orthopaedics)", "Dr. Chen (Pediatrician)", "Dr. Patel (Radiologist)"],
                            key=f"opt_doc_select_{case_id}"
                        )
                        selected_doc = raw_choice.split(" (")[0]
                        st.session_state["selected_doctor"] = selected_doc
                    else:
                        st.session_state["selected_doctor"] = None

            # Should we show the chat input?
            allow_chat_input = True
            if st.session_state.get("selected_doctor"):
                # When another doctor is selected, hide chat input per your requirement
                allow_chat_input = False
                st.info(f"✍️ {st.session_state['selected_doctor']} selected. Chat input disabled; use **Add Image Annotation** below.")

            # -----------------------------
            # Messages
            # -----------------------------
            messages = get_messages(case_id)
            with st.container():
                for msg in messages:
                    render_message_bubble(msg, st.session_state.user_name)

            # -----------------------------
            # Chat input (conditionally visible)
            # -----------------------------
            if allow_chat_input:
                message = st.chat_input("Type your message here", key=f"chat_input_{case_id}")
                if message:
                    # Add the user's message
                    add_message(case_id, st.session_state.user_name, message.strip())

                    # AI auto-reply (optional)
                    if get_ai_response:
                        with st.spinner("AI Assistant is analyzing..."):
                            time.sleep(1)
                        findings = st.session_state.get("findings", None)
                        api_key = st.session_state.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
                        ai_response = get_openai_response(message, room_data["description"], findings, api_key)
                        add_message(case_id, "Dr. AI Assistant", ai_response)

                    st.rerun()

            # -----------------------------
            # Image Annotation
            # -----------------------------
            with st.expander("Add Image Annotation", expanded=bool(st.session_state.get("selected_doctor"))):
                # Author logic:
                # - If a doctor is selected, ONLY allow that doctor (hide user option like "Dr Shah")
                # - If no doctor is selected, default to the user
                selected_doctor = st.session_state.get("selected_doctor")
                if selected_doctor:
                    st.markdown(f"Annotating as **{doctor_avatar(selected_doctor)} {selected_doctor}**")
                    author = selected_doctor
                else:
                    st.markdown(f"Annotating as **🧑‍⚕️ {st.session_state.user_name}**")
                    author = st.session_state.user_name

                annotation = st.text_area(
                    "Describe what you see in the image",
                    key=f"annot_text_{case_id}",
                    # placeholder="e.g., Rounded consolidation in RLL; no effusion; heart size normal."
                )

                if st.button("Submit Annotation", key=f"annot_submit_{case_id}"):
                    if not annotation.strip():
                        st.warning("Please enter an annotation before submitting.")
                    else:
                        add_message(case_id, author, annotation.strip(), message_type="annotation")
                        st.success(f"Annotation added as **{author}**.")
                        st.rerun()

        else:
            st.error("This case discussion no longer exists")
            if st.button("Return to Room Selection", key="return_room_select_btn"):
                del st.session_state.current_case_id
                st.rerun()

# -----------------------------
# Optional: manual room creation
# -----------------------------
def create_manual_chat_room(creator_name, case_description):
    case_id = f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return create_chat_room(case_id, creator_name, case_description)

# -----------------------------
# Run the UI
# -----------------------------
if __name__ == "__main__":
    render_chat_interface()
