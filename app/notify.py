import os
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
from config import NOTIFY_THRESHOLD_HIGH

def send_line_notify(message, token):
    if not token:
        print("  WARN: No LINE_NOTIFY_TOKEN found. Skipping notification.")
        return
    
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    
    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            print("  Success: LINE Notify sent.")
        else:
            print(f"  Error: LINE Notify failed ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"  Error: Failed to send LINE Notify: {e}")

def run_notify():
    token = os.environ.get("LINE_NOTIFY_TOKEN")
    if not token:
        print("  WARN: Skipping notification (No token).")
        return

    # Load latest dashboard data
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "processed" / "dashboard_data.csv"
    
    if not data_path.exists():
        print("  Error: dashboard_data.csv not found.")
        return
        
    df = pd.read_csv(data_path, parse_dates=["Datetime"])
    
    # Get max predicted for next 24h per province
    now = datetime.now()
    fore_24h = df[(df["Datetime"] > now.strftime("%Y-%m-%d %H:%M")) & 
                  (df["Datetime"] <= (now + pd.Timedelta(hours=24)).strftime("%Y-%m-%d %H:%M"))]
    
    if fore_24h.empty:
        print("  Info: No forecast data found for next 24h.")
        return
        
    ranking = (
        fore_24h.groupby("Province")["predicted"]
        .max()
        .reset_index()
        .sort_values("predicted", ascending=False)
    )
    
    # Filter only those above threshold
    alerts = ranking[ranking["predicted"] > NOTIFY_THRESHOLD_HIGH]
    
    if alerts.empty:
        print("  Info: No provinces exceed the notification threshold.")
        return
        
    # Build message
    today_str = datetime.now().strftime("%d %b %Y")
    msg = f"\n🚨 แจ้งเตือน PM2.5 ภาคเหนือ [{today_str}]\n\n"
    msg += "จังหวัดที่น่าเป็นห่วง (24h ข้างหน้า):\n"
    
    for _, row in alerts.iterrows():
        p = row["Province"]
        val = row["predicted"]
        emoji = "🔴" if val > 75 else "🟠"
        label = "มีผลต่อสุขภาพ" if val > 75 else "เริ่มมีผลต่อสุขภาพ"
        msg += f"{emoji} {p}: {val:.1f} µg/m³ ({label})\n"
        
    msg += "\nดูรายละเอียด:\nhttps://pm25-warning.streamlit.app" # Placeholder URL
    
    send_line_notify(msg, token)

if __name__ == "__main__":
    print("\nChecking for provinces needing notification...")
    run_notify()
    print("Done.")
