# gen_textout_font.py -- one-shot generator for kfm_textout_font.h
# Extracts the 10x20 bitmap font table (font[][20]) mechanically from the
# upstream KFM TextOut.cu (rigaya/AviSynthCUDAFilters; font taken from Donald
# Graft's decomb package, MIT) so the KDeblockOCL show==1 flag-message
# renderer (a faithful DrawText port) can be bit-exact.
# Run: python3 gen_textout_font.py <path-to-TextOut.cu>
import re
import sys

src = open(sys.argv[1], encoding="latin-1").read()
m = re.search(r"font\[\]\[20\]\s*=\s*\{(.*?)\n\};", src, re.S)
body = m.group(1)
glyphs = re.findall(r"\{([^{}]*)\}", body)
out = []
for g in glyphs:
    nums = re.findall(r"0x[0-9a-fA-F]{4}", g)
    assert len(nums) == 20, len(nums)
    out.append(nums)
assert out[0] == ["0x0000"] * 20  # entry 0 must be space
print("glyphs:", len(out))

hdr = []
hdr.append("// kfm_textout_font.h -- 10x20 bitmap font table mechanically extracted")
hdr.append("// from upstream rigaya/AviSynthCUDAFilters KFM/TextOut.cu (font[][20],")
hdr.append('// taken from Donald Graft\'s decomb package, MIT-licensed with KFM).')
hdr.append("// Used by the KDeblockOCL show==1 flag message renderer (a faithful port")
hdr.append("// of upstream DrawText/cpu_draw_text: glyph cell 10x20, luma 235<<shift,")
hdr.append("// chroma 128<<shift, unset pixels dimmed to (dst(+128uv))>>1).")
hdr.append("// Entries are indexed by (char - 32); entry count = %d (up to 'z')." % len(out))
hdr.append("// Regenerated mechanically by gen_textout_font.py; do not hand-edit.")
hdr.append("#pragma once")
hdr.append("#include <cstdint>")
hdr.append("")
hdr.append("namespace kfmtext {")
hdr.append("")
hdr.append("static const uint16_t kFont[%d][20] = {" % len(out))
for nums in out:
    hdr.append("    {" + ",".join(nums) + "},")
hdr.append("};")
hdr.append("")
hdr.append("}  // namespace kfmtext")
open(__file__.replace("gen_textout_font.py", "kfm_textout_font.h"),
     "w", newline="\n").write("\n".join(hdr) + "\n")
print("written")
