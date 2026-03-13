"""
Patches settings.yaml to add Telegram config.
Run once then delete.
"""
import yaml
from pathlib import Path

config_path = Path("config/settings.yaml")
settings = yaml.safe_load(config_path.read_text())

# Add alerts section
settings["alerts"] = {
    "telegram_enabled": True,
    "telegram_token": "8760045272:AAH2VKn6vNRFlr0pb_KVGOIl8k_v_wz0FGg",
    "telegram_chat_id": "6553286553286774",
}

config_path.write_text(yaml.dump(settings, default_flow_style=False, sort_keys=False))
print("  settings.yaml updated with Telegram config!")
