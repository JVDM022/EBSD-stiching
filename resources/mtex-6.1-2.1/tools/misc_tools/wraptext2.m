function out = wraptext2(txt,width)
%WRAPTEXT Wrap text to a maximum line length (Unicode-aware, indent-aware).
%
%   out = wraptext(txt,width) returns text where:
%     - Each original input line (split on newline) is wrapped separately.
%     - No wrapped line exceeds 'width' visible characters.
%     - Words are not split.
%     - Runs of leading whitespace in each original line are preserved
%       and re-applied to every wrapped line from that line.
%     - <a ...>...</a> fragments are treated as unbreakable words.
%
%   txt   ... string or char
%   width ... positive integer
%
%   Returns a char vector with '\n' separators.

if nargin==1
  cms = get(0,'CommandWindowSize');
  width = cms(1);
else
  assert(isscalar(width),'Width must be a scalar.')
end


% normalize to string scalar
if ~isstring(txt)
  txt = string(txt);
end
if numel(txt) ~= 1
  txt = join(txt,newline); % just in case we got a string array
end

% split on existing newlines, keep empties
lines_in = split(txt, newline);

wrapped_lines_all = strings(0,1);

for L = 1:numel(lines_in)
  original_line = lines_in(L);
  
  % If the entire line is empty, preserve it
  if strlength(original_line) == 0
    wrapped_lines_all(end+1,1) = ""; %#ok<AGROW>
    continue
  end
  
  % ---- 0) capture leading indentation (spaces/tabs etc.) ----
  % use regexp to get the leading whitespace prefix
  % (if no match -> indentPrefix = "")
  tokens = regexp(original_line, "^\s*", "match", "once");
  if isempty(tokens)
    indentPrefix = "";
  else
    indentPrefix = tokens;
  end
  
  % remove that prefix for the wrapping logic
  paragraph_body = extractAfter(original_line, strlength(indentPrefix));
  % Special case: if the line is ONLY whitespace, keep it as-is
  if strlength(paragraph_body) == 0
    wrapped_lines_all(end+1,1) = indentPrefix; %#ok<AGROW>
    continue
  end
  
  % ---- 1) protect <a ...>...</a> blocks (treat as single word) ----
  % We'll replace normal spaces inside each <a ...>...</a>...</a>
  % by a non-breaking placeholder so later "split on space"
  % won't tear them apart.
  protected_paragraph = paragraph_body;
  placeholder = char(160); % non-breaking space (nbsp)
  
  anchor_pat = "<a\b.*?>.*?</a>";
  [startIdx,endIdx] = regexp(protected_paragraph, anchor_pat, ...
    'start','end','once');
  
  % Loop to process all <a>...</a> occurrences
  while ~isempty(startIdx)
    before  = extractBetween(protected_paragraph, 1, startIdx-1);
    anchor  = extractBetween(protected_paragraph, startIdx, endIdx);
    after   = extractBetween(protected_paragraph, endIdx+1, ...
      strlength(protected_paragraph));
    
    % Replace normal spaces in the anchor block by placeholder
    anchor = replace(anchor," ",placeholder);
    
    % Reassemble
    protected_paragraph = before + anchor + after;
    
    % Search again. It's okay if we hit ones we've already
    % processed; replacing spaces doesn't affect the <a ...>...</a>
    % pattern shape.
    [startIdx,endIdx] = regexp(protected_paragraph, anchor_pat, ...
      'start','end','once');
  end
  
  % ---- 2) split into words by normal spaces ----
  words = split(protected_paragraph," ");
  
  % ---- 3) wrap these words to width ----
  current = "";
  wrapped_local = strings(0,1);
  
  for k = 1:numel(words)
    w = words(k);
    
    % visible length of w (restore placeholders for measuring)
    w_vis = replace(w,placeholder," ");
    w_len = strlength(w_vis);
    
    if current == ""
      % starting a new output line
      if w_len > width
        % word longer than width -> force it alone
        wrapped_local(end+1,1) = w_vis; %#ok<AGROW>
        current = "";
      else
        current = w;
      end
    else
      curr_vis = replace(current,placeholder," ");
      curr_len = strlength(curr_vis);
      
      if curr_len + 1 + w_len <= width
        current = current + " " + w;
      else
        % push current
        wrapped_local(end+1,1) = replace(current,placeholder," "); %#ok<AGROW>

        if w_len > width
          % word itself too long -> its own line
          wrapped_local(end+1,1) = w_vis; %#ok<AGROW>
          current = "";
        else
          current = w;
        end
      end
    end
  end
  
  if current ~= ""
    wrapped_local(end+1,1) = replace(current,placeholder," ");
  end
  
  % ---- 4) re-apply indentation prefix to EVERY wrapped line ----
  if strlength(indentPrefix) > 0
    wrapped_local = indentPrefix + wrapped_local;
  end
  
  % ---- 5) append to global result ----
  wrapped_lines_all = [wrapped_lines_all; wrapped_local]; %#ok<AGROW>
end

% Join everything with '\n'
out = strjoin(wrapped_lines_all, newline);
out = char(out); % char output is convenient for fprintf, etc.

if nargout==0
  disp(out); 
  clear out
end

end
