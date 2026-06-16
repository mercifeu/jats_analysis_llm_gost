import re
import difflib
import logging

logger = logging.getLogger("TextualGrounding")

MAX_XML_LENGTH = 1000000
MAX_ARTICLE_LENGTH = 2000000

try:
    from lxml import etree
except ImportError:
    etree = None

def normalize_for_comparison(text):
    if not text:
        return ""
    text = text.replace('–', '-').replace('—', '-').replace('−', '-')
    text = text.replace('\u00A0', ' ')
    text = text.replace('\u200B', '')
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text.strip()


def split_into_sentences(text):
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ])', text)
    return [p.strip() for p in parts if len(p.strip()) > 10]

class AtomicField:
    def __init__(self, element, root):
        self.element = element
        self.tag_name = element.tag
        self.text_content = (element.text or "").strip()
        self.attributes = dict(element.attrib) if element.attrib else {}
        self.xpath = self._build_xpath(root)
        self.length = len(self.text_content)
        self.lang = self.attributes.get('lang', None)
    
    def _build_xpath(self, root):
        path = []
        current = self.element
        while current is not None and current != root:
            tag = current.tag
            parent = current.getparent()
            if parent is not None:
                siblings = [c for c in parent if c.tag == tag]
                if len(siblings) > 1:
                    idx = siblings.index(current) + 1
                    path.append(f"{tag}[{idx}]")
                else:
                    path.append(tag)
            else:
                path.append(tag)
            current = parent
        path.reverse()
        return "/" + "/".join(path)


def collect_atomic_fields(xml_text):
    root = _parse_xml_for_grounding(xml_text)
    if root is None:
        return []
    
    fields = []
    
    def walk(element):
        if element.tag == '_DOC_':
            for child in element:
                walk(child)
            return
        text = (element.text or "").strip()
        has_children = len(list(element)) > 0
        if not has_children and text:
            fields.append(AtomicField(element, root))
        for child in element:
            walk(child)
    
    walk(root)
    return fields, root


def _parse_xml_for_grounding(xml_str):
    if not xml_str or not xml_str.strip():
        return None

    if len(xml_str) > MAX_XML_LENGTH:
        logger.warning(f"[TextualGrounding] XML слишком длинный ({len(xml_str)} > {MAX_XML_LENGTH})")
        return None
    
    cleaned = re.sub(r'```xml\s*', '', xml_str)
    cleaned = re.sub(r'```\s*', '', cleaned).strip()
    cleaned = re.sub(r'&(?![a-zA-Z]+;|#[0-9]+;|#x[0-9a-fA-F]+;)', '&amp;', cleaned)

    secure_parser = etree.XMLParser(resolve_entities=False, no_network=True)
    
    try:
        return etree.fromstring(cleaned.encode('utf-8'), parser=secure_parser)
    except etree.XMLSyntaxError:
        pass
    
    try:
        wrapped = f"<_DOC_>{cleaned}</_DOC_>"
        return etree.fromstring(wrapped.encode('utf-8'), parser=secure_parser)
    except etree.XMLSyntaxError:
        pass
    
    parser = etree.XMLParser(recover=True, encoding='utf-8', resolve_entities=False, no_network=True)
    try:
        wrapped = f"<_DOC_>{cleaned}</_DOC_>"
        return etree.fromstring(wrapped.encode('utf-8'), parser)
    except Exception:
        return None

def build_norm_to_orig_mapping(original):
    mapping = []
    last_was_space = False
    
    for i_orig in range(len(original)):
        c = original[i_orig]
        
        if c in ' \t\n\r\u00A0\u200B':
            if not last_was_space and mapping:
                mapping.append(i_orig)
                last_was_space = True
            continue
        
        if c in '–—−':
            mapping.append(i_orig)
            last_was_space = False
            continue
        
        mapping.append(i_orig)
        last_was_space = False
    
    return mapping


def normalized_text_via_mapping(original, mapping):
    chars = []
    prev_space = False
    for idx in mapping:
        c = original[idx]
        if c in ' \t\n\r\u00A0\u200B':
            if not prev_space:
                chars.append(' ')
                prev_space = True
        elif c in '–—−':
            chars.append('-')
            prev_space = False
        else:
            chars.append(c.lower())
            prev_space = False
    return ''.join(chars)


def find_exact_in_original(xml_text, original_article, mapping, norm_orig):
    norm_xml = normalize_for_comparison(xml_text)
    
    pos = norm_orig.find(norm_xml)
    if pos < 0 or pos + len(norm_xml) > len(mapping):
        return False, None
    
    start = mapping[pos]
    end = mapping[min(pos + len(norm_xml) - 1, len(mapping) - 1)] + 1
    return True, original_article[start:end]


def find_full_block_in_original(xml_text, original_article, mapping, norm_orig):
    if not xml_text or len(xml_text) < 5:
        return None, 0.0
    
    anchor = xml_text[:min(80, len(xml_text))].strip()
    anchor_norm = normalize_for_comparison(anchor)
    
    anchor_pos_in_norm = norm_orig.find(anchor_norm)
    if anchor_pos_in_norm < 0:
        full_norm = normalize_for_comparison(xml_text)
        anchor_pos_in_norm = norm_orig.find(full_norm[:min(80, len(full_norm))])
        if anchor_pos_in_norm < 0:
            return None, 0.0
    
    if anchor_pos_in_norm >= len(mapping):
        return None, 0.0
    
    orig_anchor_start = mapping[anchor_pos_in_norm]
    
    block_start = orig_anchor_start
    prev_double = original_article.rfind('\n\n', 0, orig_anchor_start)
    if prev_double >= 0:
        block_start = prev_double + 2
    else:
        prev_single = original_article.rfind('\n', 0, orig_anchor_start)
        if prev_single >= 0:
            block_start = prev_single + 1
        else:
            block_start = 0
    
    anchor_end_in_norm = anchor_pos_in_norm + len(anchor_norm)
    if anchor_end_in_norm < len(mapping):
        orig_anchor_end = mapping[anchor_end_in_norm - 1] + 1
    else:
        orig_anchor_end = orig_anchor_start + len(anchor)
    
    next_double = original_article.find('\n\n', orig_anchor_end)
    if next_double >= 0:
        block_end = next_double
    else:
        block_end = len(original_article)
    
    full_block = original_article[block_start:block_end].strip()
    
    list_marker_pattern = re.compile(
        r'^\s*(?:'
        r'[-*•·]\s+'           # -, *, •, ·
        r'|\d+[.)]\s+'         # 1., 1), 12.
        r'|\(\d+\)\s+'         # (1), (12)
        r'|[a-zа-яё]\)\s+'     # а), б), a)
        r')'
    )
    full_block = list_marker_pattern.sub('', full_block).strip()
    
    similarity = difflib.SequenceMatcher(
        None,
        normalize_for_comparison(xml_text),
        normalize_for_comparison(full_block)
    ).ratio()
    
    return full_block, similarity


def find_in_original(xml_text, original_article, mapping, norm_orig):
    if not xml_text or len(xml_text) < 3:
        return 'too_short', 0.0, None, None
    
    found_exact, same_length_fragment = find_exact_in_original(
        xml_text, original_article, mapping, norm_orig
    )
    if found_exact:
        full_block, full_sim = find_full_block_in_original(
            xml_text, original_article, mapping, norm_orig
        )
        if full_block and full_sim >= 0.95:
            return 'exact_match', 1.0, same_length_fragment, full_block
        return 'exact_match', 1.0, same_length_fragment, same_length_fragment
    
    if len(xml_text) >= 200:
        sentences = split_into_sentences(xml_text)
        if sentences:
            found_count = sum(
                1 for s in sentences 
                if find_exact_in_original(s, original_article, mapping, norm_orig)[0]
            )
            ratio = found_count / len(sentences)
            
            if ratio >= 0.50:
                full_block, _ = find_full_block_in_original(
                    xml_text, original_article, mapping, norm_orig
                )
                if ratio >= 0.90:
                    return 'exact_match', ratio, None, full_block
                elif ratio >= 0.75:
                    return 'fuzzy_match', ratio, None, full_block
                else:
                    return 'partial_match', ratio, None, full_block
    
    target_norm = normalize_for_comparison(xml_text)
    window_size = len(target_norm)
    best_similarity = 0.0
    best_pos = -1
    step = max(50, window_size // 4)
    
    for i in range(0, max(1, len(norm_orig) - window_size + 1), step):
        window = norm_orig[i:i + window_size]
        sim = difflib.SequenceMatcher(None, target_norm, window).ratio()
        if sim > best_similarity:
            best_similarity = sim
            best_pos = i
    
    if best_pos >= 0 and best_pos + window_size <= len(mapping):
        start = mapping[best_pos]
        end = mapping[best_pos + window_size - 1] + 1
        fragment = original_article[start:end]
        
        full_block, _ = find_full_block_in_original(
            xml_text, original_article, mapping, norm_orig
        )
        
        if best_similarity >= 0.95:
            return 'exact_match', best_similarity, fragment, full_block
        elif best_similarity >= 0.80:
            return 'fuzzy_match', best_similarity, fragment, full_block
        elif best_similarity >= 0.50:
            return 'partial_match', best_similarity, fragment, full_block
    
    return 'not_found', best_similarity, None, None

def classify_field(search_result):
    status, similarity, _, _ = search_result
    
    if status in ('exact_match', 'fuzzy_match', 'partial_match'):
        if similarity >= 0.95:
            return 'OK_EXACT', "Точное совпадение с оригиналом"
        else:
            return 'PARAPHRASE', f"Парафраз (sim={similarity:.2f})"
    
    return 'HALLUCINATION', "Текст не найден в оригинале"

def validate_textual_grounding(xml_text, original_article):

    if len(original_article) > MAX_ARTICLE_LENGTH:
        logger.warning(f"[TextualGrounding] Статья слишком длинная ({len(original_article)} > {MAX_ARTICLE_LENGTH})")
        return [], None
    
    fields_root = collect_atomic_fields(xml_text)
    if isinstance(fields_root, tuple):
        fields, root = fields_root
    else:
        fields = fields_root
        root = None
    
    if not fields:
        return [], root
    
    mapping = build_norm_to_orig_mapping(original_article)
    norm_orig = normalized_text_via_mapping(original_article, mapping)
    
    report = []
    for field in fields:
        search_result = find_in_original(
            field.text_content, original_article, mapping, norm_orig
        )
        classification, reason = classify_field(search_result)
        status, similarity, best_match, full_block = search_result
        
        report.append({
            'tag': field.tag_name,
            'xpath': field.xpath,
            'lang': field.lang,
            'length': field.length,
            'text_in_xml': field.text_content,
            'text_in_original': best_match,
            'recommended_replacement': full_block,
            'classification': classification,
            'reason': reason,
            'similarity': similarity,
        })
    
    return report, root

def apply_replacements(root, report):
    if root is None:
        return root, []
    
    fixes = []
    
    for item in report:
        if item['classification'] not in ('PARAPHRASE', 'HALLUCINATION'):
            continue
        
        rec = item.get('recommended_replacement')
        if not rec:
            continue
        
        xml_norm = normalize_for_comparison(item['text_in_xml'])
        rec_norm = normalize_for_comparison(rec)
        if rec_norm == xml_norm:
            continue
        
        xpath = item['xpath']
        if xpath.startswith('/_DOC_/'):
            xpath = xpath[6:]
        elif xpath.startswith('/'):
            xpath = xpath[1:]
        
        try:
            elements = root.xpath(xpath)
            if elements:
                el = elements[0]
                old_text = el.text or ""
                rec_clean = re.sub(
                    r'^\s*(?:[-*•·]\s+|\d+[.)]\s+|\(\d+\)\s+|[a-zа-яё]\)\s+)',
                    '', rec
                ).strip()
                el.text = rec_clean
                fixes.append({
                    'tag': item['tag'],
                    'xpath': xpath,
                    'old': old_text,
                    'new': rec,
                    'classification': item['classification'],
                    'similarity_before': item['similarity'],
                })
        except Exception as e:
            logger.warning(f"[TextualGrounding] XPath ошибка {xpath}: {e}", exc_info=True)
    
    return root, fixes

def pipeline_verify_and_fix(xml_str, article_text):
    if not xml_str or not article_text:
        return xml_str, []
    
    report, root = validate_textual_grounding(xml_str, article_text)
    
    if not report or root is None:
        return xml_str, []
    
    root, fixes = apply_replacements(root, report)
    
    if not fixes:
        return xml_str, []
    
    if root.tag == '_DOC_':
        parts = []
        for child in root:
            parts.append(etree.tostring(child, encoding='unicode', pretty_print=True))
        result_xml = '\n'.join(parts)
    else:
        result_xml = etree.tostring(root, encoding='unicode', pretty_print=True)
    
    return result_xml.strip(), fixes
