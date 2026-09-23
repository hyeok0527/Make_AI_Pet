from fafbseg import flywire

# 테스트할 뉴런 ID
root_id = 720575940621030000

print(f"뉴런 ID({root_id})의 기본 정보를 조회합니다...")

# 메쉬 대신 시냅스 테이블이나 메타데이터를 먼저 가져옵니다.
# (이 방식은 데이터 용량이 작아 메모리 에러가 거의 나지 않습니다.)
synapses = flywire.get_synapses(root_id)

print("시냅스 데이터 조회 성공!")
print(f"총 연결된 시냅스 수: {len(synapses)}개")
print(synapses.head())