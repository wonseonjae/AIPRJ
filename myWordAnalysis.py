from konlpy.tag import Hannanum

import re

myHannanum = Hannanum()

text = "두산에너빌리티가 엘앤에프와 손을 잡고 배터리 소재 리사이클링 사업을 가속화한다." \
       "두산에너빌리티는 엘앤에프와 ‘배터리 소재 리사이클링 사업 협력’을 위한 업무협약을 체결했다고 1일 밝혔다. 대구에 본사를 두고 있는 엘앤에프는 리튬이온 2차전지용 양극 소재 분야의 대표기업이다." \
       "이번 협약으로 엘앤에프는 양극재 생산 과정에서 발생하는 폐파우더를 제공하고, 두산에너빌리티는 폐파우더에서 리튬을 추출하는 역할을 수행하게 된다. 양극재는 배터리의 용량과 출력을 결정하는 배터리 핵심 소재다." \
       "리튬은 양극재를 구성하는 필수 원료로, 노트북과 휴대폰 등 IT 기기와 전기차 배터리에 주로 사용된다." \
       "송용진 두산에너빌리티 전략혁신부문장은 “최근 전기차 시장과 함께 배터리 시장이 빠르게 확대되면서 리튬 수요도 꾸준히 증가하고 있다”며 “엘앤에프와 협력을 통해 배터리 리사이클링 사업의 선순환 구조를 마련하고, 이를 기반으로 급속도로 성장하는 리튬 시장에 적극 참여할 것”이라고 말했다."

replace_text = re.sub("[!@$#%^&*()_+]", " ", text)

print(myHannanum.nouns(replace_text))
