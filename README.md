# math_problem_solve_2nd
자연어로 된 수학문제를 읽어 답과 풀이(python code)를 제출하는 AI 모델을 배포합니다.
    
## Requirements
tokenizers==0.10.3 
transformers==4.15.0     
json5==0.9.6     
pytorch==1.10.0   
pytorch-lightning==1.2.10   
parso==0.8.2   
numpy==1.21.2
    
## How to install
  
```
git clone https://github.com/nuualab/math_problem_solve_2nd
```
  
## How to run
```
python main.py
```
example 폴더의 example.json 파일을 읽어 추론 후 answer.json 파일을 생성 합니다.  
   
   
## Pretrained Model Download
weights 디렉토리에 저장 (산술 추론 모델)      
[arithmetic.ckpt](https://drive.google.com/file/d/1XVEiTzujs4jixTO3lgFTJPdvJZZg8eVP/view?usp=sharing, "arithmetic.ckpt")   
     
weights 디렉토리에 저장 (코드 추론 모델)      
[code.ckpt](https://drive.google.com/file/d/18Cc_s6OuAkOT67eujmrNq_a56UMPkQG9/view?usp=sharing, "code.ckpt")   
    
weights/kogpt2 디렉토리에 저장   
[pytorch_model.bin](https://drive.google.com/file/d/1oJsPsV-jIoxi3yDsKIqpIVQXYXXcvTuP/view?usp=sharing, "pytorch_model.bin")
    
## License
이 프로젝트는 Apache 2.0 라이선스를 따릅니다. 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 라이선스 전문은 LICENSE 파일에서 확인하실 수 있습니다.

*이 프로젝트는 과학기술정보통신부 인공지능산업원천기술개발사업의 지원을 통해 제작 되었습니다.*
