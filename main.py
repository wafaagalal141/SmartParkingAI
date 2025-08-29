from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import paho.mqtt.client as mqtt
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn
from supabase import create_client, Client
from typing import List, Dict, Any
import re
import os
from dotenv import load_dotenv

# ==========================
# Load environment variables
# ==========================
load_dotenv()

BROKER = "858a3089681b477597400a3b5c9ba46b.s1.eu.hivemq.cloud"
PORT = 8883
USERNAME = os.getenv("MQTT_USERNAME")
PASSWORD = os.getenv("MQTT_PASSWORD")

mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(USERNAME, PASSWORD)
mqtt_client.tls_set()

latest_ir = None
led_state = "off"

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe("home/ir")

def on_message(client, userdata, msg):
    global latest_ir
    if msg.topic == "home/ir":
        try:
            latest_ir = int(msg.payload.decode())
        except ValueError:
            latest_ir = None

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(BROKER, PORT, 60)
mqtt_client.loop_start()

# ==========================
# Tool Functions
# ==========================
def get_ir_reading(_=None):
    return str(latest_ir) if latest_ir is not None else "IR reading not available"

def turn_led_on(_=None):
    global led_state
    if led_state == "on":
        return "LED is already on"
    mqtt_client.publish("home/led", "on")
    led_state = "on"
    return "LED turned on"

def turn_led_off(_=None):
    global led_state
    if led_state == "off":
        return "LED is already off"
    mqtt_client.publish("home/led", "off")
    led_state = "off"
    return "LED turned off"

def move_servo(angle: str):
    try:
        angle_int = int(angle)
        if not 0 <= angle_int <= 180:
            return "Invalid angle. Must be between 0–180."
    except ValueError:
        return "Invalid angle. Provide an integer between 0–180."
    mqtt_client.publish("home/servo", str(angle_int))
    return f"Servo moved to {angle_int}°"

# ==========================
# Extra Smart Parking MQTT Controls
# ==========================
def control_gate(action: str):
    mqtt_client.publish("parking/control/gate", action)
    return f"Gate command sent: {action}"

def control_ceiling(action: str):
    mqtt_client.publish("parking/control/ceiling", action)
    return f"Ceiling command sent: {action}"

def control_buzzer(action: str):
    mqtt_client.publish("parking/control/buzzer", action)
    return f"Buzzer command sent: {action}"

def control_nightmode(action: str):
    mqtt_client.publish("parking/control/nightmode", action)
    return f"Night mode command sent: {action}"

def control_system(action: str):
    mqtt_client.publish("parking/control/system", action)
    return f"System command sent: {action}"

def control_slot(slot: str, action: str):
    mqtt_client.publish(f"parking/control/slot/{slot}", action)
    return f"Slot {slot} command sent: {action}"

# ==========================
# LangChain Agent
# ==========================
tools = [
    Tool(name="Get IR Reading", func=get_ir_reading, description="Get the latest IR sensor reading."),
    Tool(name="Turn LED On", func=turn_led_on, description="Turn the LED on."),
    Tool(name="Turn LED Off", func=turn_led_off, description="Turn the LED off."),
    Tool(name="Move Servo", func=move_servo, description="Move servo to a specified angle."),
    Tool(name="Control Gate", func=control_gate, description="Send gate control commands: open/close."),
    Tool(name="Control Ceiling", func=control_ceiling, description="Send ceiling control commands: open/close."),
    Tool(name="Control Buzzer", func=control_buzzer, description="Control buzzer: activate/stop/deactivate."),
    Tool(name="Control Night Mode", func=control_nightmode, description="Control night mode: on/off/auto."),
    Tool(name="Control System", func=control_system, description="System-level commands: reset/status."),
    Tool(name="Control Slot", func=lambda inp: control_slot(inp.split()[0], inp.split()[1]), description="Control individual slots. Format: '<slot_number> <action>' e.g. '1 occupied'.")
]

system_prompt = """
You are an IoT home assistant...

Available tools:
- Get IR Reading → returns latest IR sensor value.
- Turn LED On → turns the LED on.
- Turn LED Off → turns the LED off.
- Move Servo → moves the servo motor to a specified angle (0–180).
- Control Gate → open/close gate.
- Control Ceiling → open/close ceiling.
- Control Buzzer → activate/stop/deactivate buzzer.
- Control Night Mode → on/off/auto.
- Control System → reset/status.
- Control Slot → e.g., "1 occupied", "3 free", "2 auto".

Rules:
1. ALWAYS use the tools — do not explain, just call the tool.
2. For IR sensor: if none, respond exactly “IR reading not available”.
3. For LED: report if already in requested state.
4. Servo: input must be 0–180.
5. If impossible, reply with: “I cannot do that.”
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    return_only_outputs=True
)

# ==========================
# FastAPI Initialization
# ==========================
app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def get_db() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@app.get("/api/parking_slots_current", response_model=List[Dict[str, Any]])
def get_parking_slots_current(db: Client = Depends(get_db)):
    result = db.table("parking_slots_current").select("*").execute()
    return result.data

@app.get("/api/gate_status_current", response_model=List[Dict[str, Any]])
def get_gate_status_current(db: Client = Depends(get_db)):
    result = db.table("gate_status_current").select("*").execute()
    return result.data

# ==========================
# Keyword → Table Mapping
# ==========================
KEYWORD_TABLE_MAP = {
    "current parking": "parking_slots_current",
    "occupied": "parking_slots_current",
    "free now": "parking_slots_current",
    "slot status": "parking_slots_current",
    "available parking": "parking_slots_current",

    "current gate": "gate_status_current",
    "open": "gate_status_current",
    "closed": "gate_status_current",
    "gate now": "gate_status_current"
}

# ==========================
# Internal Fetch Functions
# ==========================
def fetch_parking_slots_current(db: Client):
    return db.table("parking_slots_current").select("*").execute().data

def fetch_gate_status_current(db: Client):
    return db.table("gate_status_current").select("*").execute().data

# ==========================
# Keyword Check using internal calls
# ==========================
def check_keywords_and_fetch(question: str, db: Client):
    q_lower = question.lower()
    for keyword, table in KEYWORD_TABLE_MAP.items():
        if re.search(rf"\b{keyword}\b", q_lower):
            if table == "parking_slots_current":
                data = fetch_parking_slots_current(db)
                total = len(data)
                free_slots = [d for d in data if d.get("is_occupied") is False]
                return f"There are {len(free_slots)} free parking slots right now from {total} slots."
            elif table == "gate_status_current":
                data = fetch_gate_status_current(db)
                gate = data[0].get("state") if data else None
                return f"The gate is currently {gate}."
    return None

# ==========================
# Question endpoints
# ==========================
class QuestionRequest(BaseModel):
    question: str

temp_storage = {}

@app.post("/submit-question")
def submit_question(req: QuestionRequest):
    question = req.question
    temp_storage["latest_question"] = question
    return {temp_storage["latest_question"]}

@app.get("/get-answer")
def get_answer(db: Client = Depends(get_db)):
    question = temp_storage.get("latest_question", None)
    if question is None:
        raise HTTPException(status_code=404, detail="No question submitted yet")

    q_lower = question.lower()
    control_words = ["open", "close", "activate", "deactivate", "reset", "move", "turn on", "turn off"]

    if any(word in q_lower for word in control_words):
        ai_answer = agent.run(question)
        temp_storage["latest_answer"] = ai_answer
        return {ai_answer}

    db_result = check_keywords_and_fetch(question, db)
    if db_result is not None:
        return {db_result}

    ai_answer = agent.run(question)
    temp_storage["latest_answer"] = ai_answer
    return {ai_answer}

@app.get("/")
def root():
    return {"message": "✅ API is running"}

# ==========================
# Vercel Serverless Handler
# ==========================
handler = app
