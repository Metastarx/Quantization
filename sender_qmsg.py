import requests


class SenderLayerQmsg:
    # 推送层自带配置，主控制层不感知具体实现细节
    SENDER_LAYER_QMSG_KEY = "da715edf385ef9f5729c4770c6484a4c"
    SENDER_LAYER_QQ_NUMBER = "454510781"
    SENDER_LAYER_PENDING_MESSAGES: list[str] = []
    SENDER_LAYER_MESSAGE_SEPARATOR = "\n--------------\n"

    @staticmethod
    def sender_layer_enqueue(message: str):
        if not message.strip():
            return
        SenderLayerQmsg.SENDER_LAYER_PENDING_MESSAGES.append(message)

    @staticmethod
    def sender_layer_flush() -> str:
        pending = list(SenderLayerQmsg.SENDER_LAYER_PENDING_MESSAGES)
        SenderLayerQmsg.SENDER_LAYER_PENDING_MESSAGES.clear()

        if not pending:
            return "skip: no pending messages"

        merged_message = SenderLayerQmsg.SENDER_LAYER_MESSAGE_SEPARATOR.join(pending)

        try:
            return SenderLayerQmsg.sender_layer_send(merged_message)
        except Exception as e:
            return f"failed: unexpected error: {e}"

    @staticmethod
    def sender_layer_send(message: str) -> str:
        if not message.strip():
            return "skip: empty message"

        url = f"https://qmsg.zendee.cn/send/{SenderLayerQmsg.SENDER_LAYER_QMSG_KEY}"
        data = {
            "msg": message,
            "qq": SenderLayerQmsg.SENDER_LAYER_QQ_NUMBER,
        }
        try:
            resp = requests.post(url, data=data, timeout=10)
        except requests.RequestException as e:
            return f"failed: request error: {e}"

        if resp.status_code >= 400:
            body = resp.text.replace("\n", " ").strip()
            return f"failed: HTTP {resp.status_code}, body={body[:200]}"

        return resp.text
