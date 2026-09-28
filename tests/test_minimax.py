from oracle_lite.minimax import MINIMAX_CHAT_URL, MINIMAX_M3_MODEL


def test_minimax_china_endpoint_and_model():
    assert MINIMAX_CHAT_URL == "https://api.minimaxi.com/v1/text/chatcompletion_v2"
    assert MINIMAX_M3_MODEL == "MiniMax-M3"
