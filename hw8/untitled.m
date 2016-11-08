TrainDataPred=mdl.predict(TrainDataX);

%Confusion Matrix is plotted
Target=zeros(4,size(TrainDataX,2));
Pred=zeros(4,size(TrainDataX,2));

for i=1:size(TrainDataX)
    Target(TrainDataY(i),i)=1;
    Pred(TrainDataPred(i),i)=1;
end
figure;
plotconfusion(Pred,Target);
xlabel('Predicted Label','FontWeight','bold')
ylabel('Target Label','FontWeight','bold')
