from caveclient import CAVEclient
from fafbseg import flywire

# 1단계에서 복사한 토큰 문자열을 아래 따옴표 사이에 넣으세요.
my_token = "05ddc159cea52810231b66a8dfb076fd"

# fafbseg 라이브러리를 통해 토큰을 맥북 시스템에 안전하게 저장합니다.
flywire.set_chunkedgraph_secret(my_token)

print("FlyWire API 토큰이 맥북에 성공적으로 저장되었습니다!")