"""
Armazenamento seguro das credenciais do Zabbix usando o cofre nativo do SO
(Windows Credential Manager, via a lib 'keyring') em vez de arquivo texto.
"""
import json
import keyring

SERVICE_NAME = "anomaly_detector_zabbix"
ENTRY_KEY = "connection"


def save_connection(url: str, auth_mode: str, token: str = None, user: str = None, password: str = None):
    data = {"url": url, "auth_mode": auth_mode}
    if auth_mode == "token":
        data["token"] = token
    else:
        data["user"] = user
        data["password"] = password
    keyring.set_password(SERVICE_NAME, ENTRY_KEY, json.dumps(data))


def load_connection():
    raw = keyring.get_password(SERVICE_NAME, ENTRY_KEY)
    if not raw:
        return None
    return json.loads(raw)


def clear_connection():
    try:
        keyring.delete_password(SERVICE_NAME, ENTRY_KEY)
    except keyring.errors.PasswordDeleteError:
        pass
