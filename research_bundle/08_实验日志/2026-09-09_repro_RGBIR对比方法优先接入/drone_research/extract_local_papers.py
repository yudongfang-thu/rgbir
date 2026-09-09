from pathlib import Path
from pypdf import PdfReader
import json

root=Path(__file__).resolve().parent
ws=root.parents[2]
inputs={
 'CMKD-Net':ws/'01_文献/RGB-IR_20260905新增/CMKD-Net__2026_TCSVT__Cross-Modal_KD_Oriented_Detection_Modality-Missing_Visible-Infrared.pdf',
 'CCLKD':ws/'01_文献/CCLKD__2026_GIS__Cross_Modal_Contrastive_Learning_Incomplete_Modalities.pdf',
 'CMDistill':ws/'06_历史工程_只读/LADD_public/comparison/cmdistill/paper/CMDistill__2025_JSTARS__Cross_Modal_Distillation_Framework_for_AAV_Image_Object_Detection.pdf'
}
out=root/'local_papers';out.mkdir(exist_ok=True)
receipt=[]
for name,path in inputs.items():
    reader=PdfReader(path)
    text='\n'.join('\n===== PAGE %d =====\n%s'%(i+1,p.extract_text()) for i,p in enumerate(reader.pages))
    (out/(name+'.txt')).write_text(text,encoding='utf-8')
    receipt.append(dict(name=name,source=str(path),pages=len(reader.pages),text_characters=len(text),hash_computed=False))
(out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
