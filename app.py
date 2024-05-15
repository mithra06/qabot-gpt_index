import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import ViltProcessor,ViltForQuestionAnswering

st.set_page_config(layout='wide',page_title='vqa')
processor=ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-nlvr2')
model=ViltForQuestionAnswering.from_pretrained('dandelin/vilt-b32-finetuned-nlvr2',ignore_mismatched_sizes=True)

def get_answer(img,text):
    try:
        image=Image.open(BytesIO(img)).convert('RGB')
        encoding=processor(image,text,return_tensors='pt')

        outputs=model(**encoding)
        logits=outputs.logits
        idx=logits.argmax(-1).item()
        answer=model.config.id2label[idx]

        return answer

    except Exception as e:
        return str(e)    

st.title('visual question and answering')
st.write('upload an image')
col1,col2=st.columns(2)

with col1:
    upload_file=st.file_uploader('upload image',type=['jpg','jpeg','png'])
    st.image(upload_file)

with col2:
    question=st.text_input('question')
    if upload_file and question is not None:
        if st.button('Ask question'):
            image=Image.open(upload_file)
            image_byte_array=BytesIO()
            image.save(image_byte_array,format='jpeg')
            image_bytes=image_byte_array.getvalue()

            answer=get_answer(image_bytes,question)
            st.info('your question:'+ question)
            st.success('answer: '+ answer)