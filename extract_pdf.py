import zlib, re, sys
data = open(sys.argv[1],"rb").read()
streams = re.findall(rb'stream\r?\n(.*?)\r?\nendstream', data, re.DOTALL)

# Simple parenthesized-string extractor with TJ-array space heuristic via numbers.
paren = re.compile(rb'\(((?:[^()\\]|\\.)*)\)')
showarr = re.compile(rb'\]\s*TJ')
showtj = re.compile(rb'\)\s*Tj')

def decode_text(text):
    # Walk tokens linearly: collect parenthesized strings; insert space when a
    # number < -100 appears between strings (kerning) or at Tj/TJ boundaries.
    out = bytearray()
    i = 0; n = len(text)
    while i < n:
        c = text[i:i+1]
        if c == b'(':
            # read string
            j = i+1; buf = bytearray()
            while j < n:
                cc = text[j:j+1]
                if cc == b'\\':
                    buf += text[j+1:j+2]; j += 2; continue
                if cc == b')':
                    j += 1; break
                buf += cc; j += 1
            out += buf
            i = j
        else:
            # check for negative number (kerning) -> space
            m = re.match(rb'-?\d+\.?\d*', text[i:i+12])
            if m and m.group(0).startswith(b'-') and len(m.group(0))>=4:
                out += b' '
                i += m.end()
            else:
                i += 1
    return bytes(out)

pieces=[]
for s in streams:
    try: d=zlib.decompress(s)
    except: continue
    if b'Tj' in d or b'TJ' in d:
        pieces.append(decode_text(d))
full=b'\n'.join(pieces)
s=full.decode('latin-1',errors='replace')
s=re.sub(r'[ ]{2,}',' ',s)
open(sys.argv[2],'w').write(s)
print("LEN", len(s))
