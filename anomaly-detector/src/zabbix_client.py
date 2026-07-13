"""
Cliente simples para a API JSON-RPC do Zabbix.

Suporta dois metodos de autenticacao:
- API Token (recomendado, Zabbix 5.4+): usado direto no header Authorization.
- Usuario/senha (metodo classico): faz login via 'user.login' e usa o token
  de sessao retornado.
"""
import time
import requests


class ZabbixAPIError(Exception):
    pass


class ZabbixClient:
    def __init__(self, url: str, token: str = None, user: str = None, password: str = None, timeout: int = 15):
        if not url:
            raise ValueError("URL do Zabbix nao informada")
        base = url.rstrip("/")
        if not base.endswith("api_jsonrpc.php"):
            base = base + "/api_jsonrpc.php"
        self.endpoint = base
        self.timeout = timeout
        self._id = 0
        self.auth_token = None

        if token:
            self.auth_token = token
        elif user and password:
            self.auth_token = self._login(user, password)
        else:
            raise ValueError("Informe um token de API ou usuario/senha")

    # ---------------------------------------------------------------
    def _next_id(self):
        self._id += 1
        return self._id

    def _request(self, method: str, params: dict = None, use_auth: bool = True):
        headers = {"Content-Type": "application/json-rpc"}
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._next_id(),
        }

        if use_auth and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            err = data["error"]
            # Fallback: algumas versoes mais antigas do Zabbix nao aceitam o
            # token via header e esperam o campo "auth" no corpo da requisicao.
            if use_auth and self.auth_token and "auth" not in payload["params"]:
                payload["params"] = dict(payload["params"])
                payload["auth"] = self.auth_token
                resp2 = requests.post(self.endpoint, json=payload, timeout=self.timeout)
                data2 = resp2.json()
                if "error" not in data2:
                    return data2["result"]
            raise ZabbixAPIError(f"{err.get('message')}: {err.get('data')}")

        return data["result"]

    def _login(self, user: str, password: str) -> str:
        result = self._request(
            "user.login",
            {"username": user, "password": password},
            use_auth=False,
        )
        return result

    # ---------------------------------------------------------------
    def test_connection(self) -> str:
        """Retorna a versao da API se a conexao/credenciais estiverem ok."""
        return self._request("apiinfo.version", {}, use_auth=False)

    def get_hosts(self):
        """Lista hosts monitorados: [{hostid, host, name}, ...]"""
        return self._request(
            "host.get",
            {"output": ["hostid", "host", "name"], "sortfield": "name"},
        )

    def get_items(self, host_id: str):
        """Lista itens numericos de um host (candidatos a metricas)."""
        items = self._request(
            "item.get",
            {
                "hostids": host_id,
                "output": ["itemid", "name", "key_", "value_type", "units"],
                "sortfield": "name",
            },
        )
        # value_type: 0=float, 3=unsigned int -> os unicos uteis para series numericas
        return [i for i in items if i.get("value_type") in ("0", "3")]

    def get_item_status(self, item_id: str):
        """
        Retorna {itemid, lastclock, lastvalue} de um item - usado para
        diagnostico rapido: se o Zabbix diz que a ultima coleta foi ha
        pouco tempo mas o history.get nao acha nada nesse intervalo,
        e sinal de relogio dessincronizado entre esta maquina e o Zabbix.
        """
        result = self._request(
            "item.get",
            {"itemids": item_id, "output": ["itemid", "lastclock", "lastvalue"]},
        )
        return result[0] if result else None

    def get_history(self, item_id: str, value_type: str, time_from: int, time_till: int = None):
        """
        Busca o historico de um item.
        value_type: '0' (float) ou '3' (unsigned int) - vem de get_items().
        time_from/time_till: timestamps unix.
        """
        params = {
            "itemids": item_id,
            "history": int(value_type),
            "time_from": time_from,
            "output": "extend",
            "sortfield": "clock",
            "sortorder": "ASC",
        }
        if time_till:
            params["time_till"] = time_till
        return self._request("history.get", params)

    @staticmethod
    def days_ago_timestamp(days: int) -> int:
        return int(time.time()) - days * 86400
