import os
from openai import OpenAI
from dotenv import load_dotenv

# --- CONFIGURATION ---
TRAINING_FILE = "./data/training_data.jsonl"
BASE_MODEL = "gpt-4o-mini"  # A proven, reliable base for fine-tuning
# ---------------------

def launch_training():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("FATAL: OPENAI_API_KEY not found in .env file. Please add it.")
        return

    client = OpenAI(api_key=api_key)

    # 1. Upload the file
    print(f"Uploading training file: {TRAINING_FILE}...")
    try:
        training_file = client.files.create(
            file=open(TRAINING_FILE, "rb"),
            purpose="fine-tune"
        )
        print(f"File uploaded successfully. File ID: {training_file.id}")
    except Exception as e:
        print(f"Error uploading file: {e}")
        return

    # 2. Start the fine-tuning job
    print(f"Starting fine-tuning job on base model: {BASE_MODEL}...")
    try:
        job = client.fine_tuning.jobs.create(
            training_file=training_file.id,
            model=BASE_MODEL
        )
        print(f"Fine-tuning job started. Job ID: {job.id}, Status: {job.status}")
        
        # 3. Stream events to monitor progress (This is the professional way)
        print("\nStreaming job events... (This will take a while, press Ctrl+C to stop streaming)")
        events = client.fine_tuning.jobs.stream_events(job.id)
        for event in events:
            print(f"  -> {event.message}")
            if event.data and event.data.get("step_count"):
                 total_steps = event.data.get("total_steps") or "unknown"
                 train_step = event.data.get("step_count", {}).get("train") or "unknown"
                 print(f"     Progress: Step {train_step} / {total_steps}")

        # 4. Get the final job status
        final_job = client.fine_tuning.jobs.retrieve(job.id)
        if final_job.status == "succeeded":
            print("\n--- 🚀 FINE-TUNING SUCCEEDED ---")
            print(f"Your new custom model ID is:")
            print(f"  {final_job.fine_tuned_model}  ")
            print("\nSave this model ID. We'll use it in the next step.")
        else:
            print(f"\n--- ❌ FINE-TUNING FAILED ---")
            print(f"Final job status: {final_job.status}")
            if final_job.error:
                print(f"Details: {final_job.error.message}")

    except Exception as e:
        print(f"Error starting or monitoring job: {e}")

if __name__ == "__main__":
    launch_training()
