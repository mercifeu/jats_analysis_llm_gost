from lxml import etree
from collections import OrderedDict, Counter
import re
import copy
import json
import logging

logger = logging.getLogger("HypothesisSchema")

MAX_XML_LENGTH = 1000000

try:
    from textual_grounding import pipeline_verify_and_fix as textual_pipeline_verify_and_fix
    TEXTUAL_GROUNDING_AVAILABLE = True
except ImportError:
    TEXTUAL_GROUNDING_AVAILABLE = False
    logger.warning("[Schema] textual_grounding.py не найден, textual verification отключена")

def _parse_xml_robust(xml_str):
    if not xml_str or not xml_str.strip():
        return None

    if len(xml_str) > MAX_XML_LENGTH:
        logger.warning(f"[Schema] XML слишком длинный ({len(xml_str)} > {MAX_XML_LENGTH})")
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
        root = etree.fromstring(wrapped.encode('utf-8'), parser)
        if root is not None and len(root) > 0:
            return root
    except Exception:
        pass
        
    return None

def infer_schema(example_xml_str):
    root = _parse_xml_robust(example_xml_str)
    if root is None:
        logger.warning("[Schema] Не удалось извлечь элементы из примера XML")
        return None

    is_fragment = (root.tag == '_DOC_')

    schema = {
        'tags': OrderedDict(),
        'hierarchy': {}
    }

    def walk(element, parent_tag=None, depth=0):
        tag = element.tag
        if tag == '_DOC_': 
            for child in element:
                walk(child, parent_tag=None, depth=0)
            return

        if tag not in schema['tags']:
            schema['tags'][tag] = {
                'attributes': dict(element.attrib),
                'has_text': bool(element.text and element.text.strip()),
                'parents': set(),
                'count': 0,
                'depth': depth,
                'children_order': [],
            }

        schema['tags'][tag]['count'] += 1
        if parent_tag:
            schema['tags'][tag]['parents'].add(parent_tag)

        if parent_tag not in schema['hierarchy']:
            schema['hierarchy'][parent_tag] = []
        schema['hierarchy'][parent_tag].append(tag)

        children_tags = [child.tag for child in element]
        schema['tags'][tag]['children_order'] = children_tags

        for child in element:
            walk(child, parent_tag=tag, depth=depth + 1)

    walk(root)

    for tag, info in schema['tags'].items():
        info['repeatable'] = info['count'] > 1
        info['parents'] = list(info['parents'])

    return schema


def schema_to_description(schema):
    if not schema:
        return ""
    
    lines = ["Ожидаемые теги и их атрибуты:"]
    
    for tag, info in schema['tags'].items():
        attrs_str = ""
        if info['attributes']:
            attrs_str = " " + " ".join(
                f'{k}="{v}"' for k, v in info['attributes'].items()
            )
        repeat = " (может повторяться)" if info['repeatable'] else ""
        text_note = " [текстовое содержимое]" if info['has_text'] else ""
        parent_note = f" (внутри <{info['parents'][0]}>)" if info['parents'] else " [корень]"
        
        lines.append(
            f"  <{tag}{attrs_str}>{text_note}{repeat}{parent_note}"
        )
    
    return "\n".join(lines)

def validate_structure(generated_xml_str, schema):
    problems = []

    gen_root = _parse_xml_robust(generated_xml_str)
    if gen_root is None:
        problems.append({
            'type': 'xml_syntax_error',
            'message': 'Сгенерированный XML не парсится вообще',
            'severity': 'critical',
            'fixable_by_code': False
        })
        return problems

    gen_tag_info = {}
    def collect(el, parent=None):
        tag = el.tag
        if tag == '_DOC_':
            for child in el:
                collect(child, parent=None)
            return

        if tag not in gen_tag_info:
            gen_tag_info[tag] = {
                'count': 0,
                'instances': [],
                'attributes_found': set()
            }
        gen_tag_info[tag]['count'] += 1
        gen_tag_info[tag]['instances'].append({
            'element': el,
            'parent': parent.tag if parent is not None and parent.tag != '_DOC_' else None
        })
        for attr_name in el.attrib:
            gen_tag_info[tag]['attributes_found'].add(attr_name)
        for child in el:
            collect(child, parent=el)
    collect(gen_root)

    for tag, expected_info in schema['tags'].items():
        if tag not in gen_tag_info:
            problems.append({
                'type': 'missing_tag',
                'tag': tag,
                'expected_attrs': expected_info['attributes'],
                'expected_parent': expected_info['parents'][0] if expected_info['parents'] else None,
                'has_text': expected_info['has_text'],
                'severity': 'critical',
                'fixable_by_code': True,
                'needs_llm': expected_info['has_text']
            })
        elif expected_info['repeatable'] and gen_tag_info[tag]['count'] < expected_info['count']:
            problems.append({
                'type': 'insufficient_count',
                'tag': tag,
                'expected_count': expected_info['count'],
                'actual_count': gen_tag_info[tag]['count'],
                'severity': 'warning',
                'fixable_by_code': False,
                'needs_llm': True
            })

    for tag in gen_tag_info:
        if tag not in schema['tags']:
            problems.append({
                'type': 'extra_tag',
                'tag': tag,
                'severity': 'warning',
                'fixable_by_code': True,
                'needs_llm': False
            })

    for tag, expected_info in schema['tags'].items():
        if tag in gen_tag_info:
            for attr_name, attr_value in expected_info['attributes'].items():
                if attr_name not in gen_tag_info[tag]['attributes_found']:
                    problems.append({
                        'type': 'missing_attribute',
                        'tag': tag,
                        'attribute': attr_name,
                        'expected_value': attr_value,
                        'severity': 'critical',
                        'fixable_by_code': True,
                        'needs_llm': False
                    })

    return problems


def fix_structure(generated_xml_str, schema, problems):
    root = _parse_xml_robust(generated_xml_str)
    if root is None:
        return generated_xml_str, problems, []

    if root.tag != '_DOC_':
        wrapper = etree.Element('_DOC_')
        wrapper.append(root)
        root = wrapper

    remaining = []
    fixes = []

    for problem in problems:
        ptype = problem['type']

        if ptype == 'extra_tag':
            for el in root.xpath(f'//{problem["tag"]}'):
                parent = el.getparent()
                if parent is not None:
                    parent.remove(el)
            fixes.append(f"Удалён лишний тег <{problem['tag']}>")

        elif ptype == 'missing_attribute':
            for el in root.xpath(f'//{problem["tag"]}'):
                el.attrib[problem['attribute']] = problem['expected_value']
            fixes.append(
                f"Добавлен атрибут {problem['attribute']}=\"{problem['expected_value']}\" "
                f"к <{problem['tag']}>"
            )

        elif ptype == 'missing_tag':
            if problem['fixable_by_code']:
                parent_tag = problem.get('expected_parent')
                parent_el = None
                
                if parent_tag:
                    found = root.xpath(f'//{parent_tag}')
                    if found:
                        parent_el = found[0]
                else:
                    parent_el = root
                
                if parent_el is not None:
                    new_el = etree.Element(problem['tag'])
                    for attr_name, attr_value in problem['expected_attrs'].items():
                        new_el.attrib[attr_name] = attr_value
                    if problem.get('has_text'):
                        new_el.text = ""
                    parent_el.append(new_el)
                    fixes.append(f"Добавлен пустой тег <{problem['tag']}>")
                    
                    if problem.get('needs_llm'):
                        remaining.append(problem)
                else:
                    remaining.append(problem)
            else:
                remaining.append(problem)

        elif ptype == 'insufficient_count':
            remaining.append(problem)

        elif ptype == 'xml_syntax_error':
            remaining.append(problem)

    if root.tag == '_DOC_':
        parts = []
        for child in root:
            parts.append(etree.tostring(child, encoding='unicode', pretty_print=True))
        fixed_xml = '\n'.join(parts)
    else:
        fixed_xml = etree.tostring(root, encoding='unicode', pretty_print=True)
    
    return fixed_xml.strip(), remaining, fixes

def build_targeted_repair_prompt(article_text, missing_fields, schema):
    fields_desc = []
    for problem in missing_fields:
        tag = problem['tag']
        tag_info = schema['tags'].get(tag, {})
        attrs = tag_info.get('attributes', {})
        attrs_str = " " + " ".join(f'{k}="{v}"' for k, v in attrs.items()) if attrs else ""
        parent = problem.get('expected_parent', 'корень')
        fields_desc.append(f"  <{tag}{attrs_str}> (внутри <{parent}>)")

    text_for_extraction = article_text[:6000]

    return f"""Извлеки из текста научной статьи данные только для перечисленных полей.

Нужные поля:
{chr(10).join(fields_desc)}

Правила:
- Бери данные ТОЛЬКО из текста статьи. Не придумывай.
- Если данных нет в тексте — напиши "НЕ НАЙДЕНО"
- Выдай только XML-фрагменты, без пояснений

Текст статьи:
{text_for_extraction}"""

def compute_metrics(generated_xml_str, reference_xml_str):
    gen_root = _parse_xml_robust(generated_xml_str)
    if gen_root is None:
        return {
            'tag_precision': 0, 'tag_recall': 0, 'tag_f1': 0,
            'attr_accuracy': 0, 'text_accuracy': 0,
            'xml_valid': False, 'error': 'Cannot parse generated XML'
        }

    ref_root = _parse_xml_robust(reference_xml_str)
    if ref_root is None:
        return {'error': 'Cannot parse reference XML'}

    def get_all_tags(root):
        tags = set()
        def walk(el):
            if el.tag != '_DOC_': tags.add(el.tag)
            for child in el: walk(child)
        walk(root)
        return tags

    gen_tags = get_all_tags(gen_root)
    ref_tags = get_all_tags(ref_root)

    common = gen_tags & ref_tags
    tag_precision = len(common) / len(gen_tags) if gen_tags else 0
    tag_recall = len(common) / len(ref_tags) if ref_tags else 0
    tag_f1 = (2 * tag_precision * tag_recall / (tag_precision + tag_recall)
              if (tag_precision + tag_recall) else 0)

    def get_all_attrs(root):
        result = {}
        def walk(el):
            tag = el.tag
            if tag == '_DOC_': 
                for child in el: walk(child)
                return
            if tag not in result: result[tag] = []
            result[tag].append(dict(el.attrib))
            for child in el: walk(child)
        walk(root)
        return result

    gen_attrs = get_all_attrs(gen_root)
    ref_attrs = get_all_attrs(ref_root)

    attr_matches, attr_total = 0, 0
    for tag in common:
        ref_tag_attrs = ref_attrs.get(tag, [{}])
        gen_tag_attrs = gen_attrs.get(tag, [{}])
        for i, ref_instance in enumerate(ref_tag_attrs):
            gen_instance = gen_tag_attrs[i] if i < len(gen_tag_attrs) else {}
            for attr_name, ref_val in ref_instance.items():
                attr_total += 1
                if gen_instance.get(attr_name) == ref_val:
                    attr_matches += 1

    attr_accuracy = attr_matches / attr_total if attr_total else 1.0

    def get_text_values(root):
        values = {}
        def walk(el, path=""):
            if el.tag == '_DOC_':
                for child in el: walk(child, path)
                return
            current = f"{path}/{el.tag}"
            if el.text and el.text.strip():
                values[current] = el.text.strip()[:200]
            for child in el: walk(child, current)
        walk(root)
        return values

    gen_values = get_text_values(gen_root)
    ref_values = get_text_values(ref_root)

    common_paths = set(gen_values.keys()) & set(ref_values.keys())
    text_matches = 0
    for path in common_paths:
        gv = gen_values[path].lower().strip()
        rv = ref_values[path].lower().strip()
        if gv == rv or rv in gv or gv in rv:
            text_matches += 1

    text_total = len(ref_values)
    text_accuracy = text_matches / text_total if text_total else 1.0

    return {
        'tag_precision': round(tag_precision, 3),
        'tag_recall': round(tag_recall, 3),
        'tag_f1': round(tag_f1, 3),
        'attr_accuracy': round(attr_accuracy, 3),
        'text_accuracy': round(text_accuracy, 3),
        'missing_tags': sorted(ref_tags - gen_tags),
        'extra_tags': sorted(gen_tags - ref_tags),
        'xml_valid': True,
        'expected_tags': len(ref_tags),
        'generated_tags': len(gen_tags),
    }

def pipeline_validate_and_fix(generated_xml_str, example_xml_str, article_text=None, llm_generate_fn=None):
    logger.info(f"[Schema Pipeline] Начало: {len(generated_xml_str)} chars")
    
    schema = infer_schema(example_xml_str)
    if not schema:
        logger.warning("[Schema Pipeline] Не удалось извлечь схему, возврат как есть")
        return generated_xml_str
    
    logger.info(f"[Schema Pipeline] Схема: {len(schema['tags'])} уникальных тегов")
    
    problems = validate_structure(generated_xml_str, schema)
    logger.info(f"[Schema Pipeline] Найдено проблем: {len(problems)}")
    for p in problems:
        logger.debug(f"  - [{p['severity']}] {p['type']}: <{p.get('tag', '')}>")
    
    if not problems:
        logger.info("[Schema Pipeline] Проблем не найдено")
        return generated_xml_str
    
    fixed_xml, remaining_problems, fixes = fix_structure(generated_xml_str, schema, problems)
    logger.info(f"[Schema Pipeline] Автоисправлений: {len(fixes)}")
    for f in fixes:
        logger.debug(f"  {f}")
    
    content_problems = [p for p in remaining_problems if p.get('needs_llm')]
    
    if content_problems and article_text and llm_generate_fn:
        logger.info(f"[Schema Pipeline] Точечных LLM-запросов: {len(content_problems)}")
        
        targeted_prompt = build_targeted_repair_prompt(article_text, content_problems, schema)
        targeted_result = llm_generate_fn(targeted_prompt, max_tokens=2048)

        if targeted_result is None or not isinstance(targeted_result, str):
            logger.warning("[Schema Pipeline] LLM вернул пустой ответ для точечного исправления")
        else:
            fixed_xml = _inject_targeted_results(fixed_xml, targeted_result, content_problems, schema)
        logger.info(f"[Schema Pipeline] После точечного исправления: {len(fixed_xml)} chars")
    
    if TEXTUAL_GROUNDING_AVAILABLE and article_text:
        logger.info(f"[Schema Pipeline] Этап 6: Проверка текстового наполнения")
        try:
            fixed_xml, textual_fixes = textual_pipeline_verify_and_fix(fixed_xml, article_text)
            logger.info(f"[Schema Pipeline] Текстовых исправлений: {len(textual_fixes)}")
            for fix in textual_fixes:
                logger.debug(
                    f"  <{fix['tag']}> ({fix['classification']}, "
                    f"sim={fix['similarity_before']:.2f})"
                )
        except Exception as e:
            logger.error(f"[Schema Pipeline] Текстуальное наполнение не удалось: {e}", exc_info=True)
    
    if '<_DOC_>' in fixed_xml or '</_DOC_>' in fixed_xml:
        fixed_xml = re.sub(r'</?_DOC_>\s*', '', fixed_xml).strip()
    
    return fixed_xml


def _inject_targeted_results(xml_str, llm_response, content_problems, schema):
    root = _parse_xml_robust(xml_str)
    if root is None:
        return xml_str

    list_marker_pattern = re.compile(r'^\s*(?:[-*•]\s+|\d+[.)]\s+|\(\d+\)\s+)+')

    for problem in content_problems:
        tag = problem['tag']
        for el in root.xpath(f'//{tag}'):
            if not (el.text and el.text.strip()):
                pattern = rf'<{tag}[^>]*>(.*?)</{tag}>'
                match = re.search(pattern, llm_response, re.DOTALL)
                if match:
                    extracted = match.group(1).strip()
                    
                    extracted = list_marker_pattern.sub('', extracted).strip()
                    
                    if len(extracted) > 2:
                        if (extracted.startswith('"') and extracted.endswith('"')) or \
                           (extracted.startswith("'") and extracted.endswith("'")) or \
                           (extracted.startswith('«') and extracted.endswith('»')):
                            extracted = extracted[1:-1].strip()
                    
                    if extracted and extracted.upper() != "НЕ НАЙДЕНО":
                        el.text = extracted
                        logger.debug(f"Заполнен <{tag}>: {extracted[:50]}...")
                break

    if root.tag == '_DOC_':
        parts = []
        for child in root:
            parts.append(etree.tostring(child, encoding='unicode', pretty_print=True))
        result_xml = '\n'.join(parts)
    else:
        result_xml = etree.tostring(root, encoding='unicode', pretty_print=True)
    
    return result_xml.strip()
