function grains = reduceBoundary(grains)
%
%
%  1 2     1
%  2 3     0
%  3 4     1 
%  4 5     0
%  10 11   1
%  11 12
%  12 13
%

F = grains.boundary.F;

tPId = grains.triplePoints.id;

canBeRemoved = [ F(2:end,1) == F(1:end-1,2); false ];

canBeRemoved = canBeRemoved & ~ismember(F(:,2),tPId);


next = true;
for k = 1:length(canBeRemoved)
  tmp = ~canBeRemoved(k) || ~next;
  canBeRemoved(k) = canBeRemoved(k) & next;
  next = tmp;
end

canBeRemoved2 = [false;canBeRemoved(1:end-1)];

F(canBeRemoved2,1) = F(canBeRemoved,1);
grains.boundary.F = F;

grains.boundary(canBeRemoved) = [];

