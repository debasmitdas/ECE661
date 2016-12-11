function feat = computeFeature(I, r, c)

feat=zeros(166000,1);

%extract Horizontal features 
cnt = 1;
for h = 1:20
for w = 1:20
for i = 1:21-h
for j = 1:41-2*w
rect1=[i,j,w,h];
rect2=[i,j+w,w,h];
feat(cnt)=sumBox(I, rect2)-sumBox(I, rect1);
cnt=cnt+1;
end
end
end
end
for h = 1:10
for w = 1:40
for i = 1:21-2*h
for j = 1:41-w
rect1=[i,j,w,h];
rect2=[i+h,j,w,h];
feat(cnt)=sumBox(I, rect1)-sumBox(I, rect2);
cnt=cnt+1;
end
end
end
end