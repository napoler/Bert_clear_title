from tkitMarker_bert import Marker

text="柯基犬是一个小短腿"
word="柯基犬"
#加载预测描述
pred=Marker(model_path="./model")
model,tokenizer=pred.load_model()
pall=pred.pre(word,text,model,tokenizer)
print(word,pall)
