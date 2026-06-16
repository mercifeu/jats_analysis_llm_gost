import os
import json
import hashlib
import re
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from difflib import SequenceMatcher
import logging
from lxml import etree as ET
import numpy as np

logger = logging.getLogger("RAG_Engine")

MAX_ARTICLE_LENGTH = 1000000
MAX_XML_LENGTH = 500000

try:
    from ai_engine import generate_response, check_connection
    HAS_AI_ENGINE = True
except ImportError:
    HAS_AI_ENGINE = False

try:
    from textual_grounding import pipeline_verify_and_fix
    HAS_TEXTUAL_GROUNDING = True
except ImportError:
    HAS_TEXTUAL_GROUNDING = False

RAG_DB_PATH = os.path.join("memory", "rag_database.json")
METADATA_CACHE_PATH = os.path.join("memory", "metadata_cache.json")

TOP_K = 3
USE_LLM_JUDGE = True
VERIFY_METADATA = True
AUTO_MIGRATE = True
ENABLE_LLM_METADATA = True

def get_rag_db():
    if not os.path.exists(RAG_DB_PATH):
        return {"examples": []}
    try:
        with open(RAG_DB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "examples" not in data:
                return {"examples": []}
            return data
    except Exception as e:
        logger.error(f"[RAG] Ошибка чтения базы: {e}", exc_info=True)
        return {"examples": []}


def save_rag_db(db):
    os.makedirs(os.path.dirname(RAG_DB_PATH), exist_ok=True)
    with open(RAG_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def compute_id(text):
    return hashlib.sha256(text[:3000].encode('utf-8')).hexdigest()[:16]


def add_example(before, after, source_template=None):
    if len(before) > MAX_ARTICLE_LENGTH:
        logger.warning(f"[RAG] Статья слишком длинная ({len(before)} > {MAX_ARTICLE_LENGTH})")
        return False, f"Статья слишком длинная (максимум {MAX_ARTICLE_LENGTH} символов)", ""
   
    if len(after) > MAX_XML_LENGTH:
        logger.warning(f"[RAG] XML слишком длинный ({len(after)} > {MAX_XML_LENGTH})")
        return False, f"XML слишком длинный (максимум {MAX_XML_LENGTH} символов)", ""
    if not before or not after:
        return False, "Пустые before/after", ""
    
    db = get_rag_db()
    example_id = compute_id(before)
    
    for ex in db["examples"]:
        if ex["id"] == example_id:
            return False, "Пример уже есть в базе", example_id
    
    new_example = {
        "id": example_id,
        "before": before,
        "after": after,
        "source_template": source_template,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": 1,
        "metadata": None,
        "metadata_hash": None,
    }
    
    db["examples"].append(new_example)
    save_rag_db(db)
    
    logger.info(f"[RAG] Добавлен новый пример (id={example_id[:8]}..., база: {len(db['examples'])} шт.)") 
    
    return True, "Успешно добавлено", example_id


def remove_example(example_id):
    db = get_rag_db()
    initial_count = len(db["examples"])
    db["examples"] = [ex for ex in db["examples"] if ex["id"] != example_id]
    
    if len(db["examples"]) < initial_count:
        save_rag_db(db)
        logger.info(f"[RAG] Удалён пример {example_id[:8]}...")
        return True
    return False


def get_db_stats():
    db = get_rag_db()
    examples_with_metadata = sum(1 for ex in db["examples"] if ex.get("metadata"))
    return {
        "total_examples": len(db["examples"]),
        "examples_with_metadata": examples_with_metadata,
        "templates_used": len(set(ex.get("source_template") for ex in db["examples"] if ex.get("source_template"))),
        "oldest": min((ex.get("timestamp") for ex in db["examples"]), default=None),
        "newest": max((ex.get("timestamp") for ex in db["examples"]), default=None),
        "metadata_cache_size": _get_metadata_cache_size(),
    }


def remove_text_tag_from_xml(xml_text, tag_name="text"):
    if not xml_text:
        return xml_text
    pattern = rf'<{tag_name}[^>]*>.*?</{tag_name}>'
    result = re.sub(pattern, '', xml_text, flags=re.DOTALL)
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    return result.strip()

def _is_llm_available() -> bool:
    if not HAS_AI_ENGINE:
        return False
    try:
        return check_connection()
    except Exception:
        return False


def _get_metadata_cache_size() -> int:
    if not os.path.exists(METADATA_CACHE_PATH):
        return 0
    try:
        with open(METADATA_CACHE_PATH, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            return len(cache)
    except Exception:
        return 0

def _compute_article_hash(article_text: str) -> str:
    normalized = re.sub(r'\s+', ' ', article_text.strip())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


def _load_metadata_cache() -> Dict[str, Dict]:
    if not os.path.exists(METADATA_CACHE_PATH):
        return {}
    try:
        with open(METADATA_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[RAG Cache] Ошибка загрузки кэша: {e}")
        return {}


def _save_metadata_cache(cache: Dict[str, Dict]):
    try:
        os.makedirs(os.path.dirname(METADATA_CACHE_PATH), exist_ok=True)
        with open(METADATA_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[RAG Cache] Ошибка сохранения кэша: {e}")


def _get_cached_metadata(article_text: str) -> Optional[Dict]:
    article_hash = _compute_article_hash(article_text)
    cache_key = f"article_{article_hash}"
    cache = _load_metadata_cache()
    return cache.get(cache_key)


def _store_cached_metadata(article_text: str, metadata: Dict):
    article_hash = _compute_article_hash(article_text)
    cache_key = f"article_{article_hash}"
    cache = _load_metadata_cache()
    cache[cache_key] = metadata
    _save_metadata_cache(cache)

def _safe_text(elem) -> str:
    if elem is None:
        return ''
    return (elem.text or '').strip()


def _find_all_by_tag(root, tag_name: str) -> List:
    results = []
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == tag_name:
            results.append(elem)
    return results


def _parse_jats_xml(root: ET.Element) -> Dict[str, Any]:
    metadata = {}
    
    for elem in root.iter():
        if elem.tag.endswith('article-id') or elem.tag == 'article-id':
            if elem.get('pub-id-type') == 'doi':
                metadata['doi'] = _safe_text(elem)
                break
    metadata.setdefault('doi', '')
    
    metadata['udc'] = ''
    metadata['bbk'] = ''
    
    for subj in _find_all_by_tag(root, 'subj-group'):
        subject_elem = None
        for child in subj:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 'subject':
                subject_elem = child
                break
        
        if subject_elem is not None:
            text = _safe_text(subject_elem)
            if 'УДК' in text or 'udc' in text.lower():
                udc_match = re.search(r'УДК\s*([\d\.\:\-\+\(\)\'"]+)', text, re.IGNORECASE)
                metadata['udc'] = udc_match.group(1).strip() if udc_match else text.replace('УДК', '').strip()
            elif 'ББК' in text or 'bbk' in text.lower():
                bbk_match = re.search(r'ББК\s*([\w\.\(\)]+)', text, re.IGNORECASE)
                metadata['bbk'] = bbk_match.group(1).strip() if bbk_match else text.replace('ББК', '').strip()
    
    metadata['title_ru'] = ''
    for elem in _find_all_by_tag(root, 'article-title'):
        metadata['title_ru'] = _safe_text(elem)
        break
    
    metadata['authors'] = []
    for contrib in _find_all_by_tag(root, 'contrib'):
        if contrib.get('contrib-type') != 'author':
            continue
        author = {}
        name_parts = []
        for child in contrib.iter():
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 'surname':
                name_parts.append(_safe_text(child))
            elif tag == 'given-names':
                name_parts.append(_safe_text(child))
            elif tag == 'contrib-id' and child.get('contrib-id-type') == 'orcid':
                author['orcid'] = _safe_text(child)
        author['name'] = ' '.join([p for p in name_parts if p])
        author.setdefault('orcid', '')
        if author['name']:
            metadata['authors'].append(author)
    
    metadata['affiliations'] = []
    for aff in _find_all_by_tag(root, 'aff'):
        aff_data = {'id': aff.get('id', ''), 'institution': '', 'city': '', 'country': ''}
        for child in aff.iter():
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 'institution':
                aff_data['institution'] = _safe_text(child)
            elif tag == 'addr-line':
                aff_data['city'] = _safe_text(child)
            elif tag == 'country':
                aff_data['country'] = _safe_text(child)
        metadata['affiliations'].append(aff_data)
        
    metadata['abstract_ru'] = ''
    for elem in _find_all_by_tag(root, 'abstract'):
        text_parts = []
        for p in elem.iter():
            tag = p.tag.split('}')[-1] if '}' in p.tag else p.tag
            if tag == 'p':
                text_parts.append(_safe_text(p))
        metadata['abstract_ru'] = ' '.join(text_parts)
        break
    
    metadata['keywords_ru'] = []
    for kwd_group in _find_all_by_tag(root, 'kwd-group'):
        keywords = []
        for kwd in _find_all_by_tag(kwd_group, 'kwd'):
            text = _safe_text(kwd)
            if text:
                keywords.append(text)
        if keywords:
            metadata['keywords_ru'] = keywords
            break
    
    for elem in _find_all_by_tag(root, 'funding-statement'):
        metadata['funding'] = _safe_text(elem)
        break
    metadata.setdefault('funding', '')

    refs = _find_all_by_tag(root, 'ref')
    metadata['references_count'] = len(refs)
    
    return metadata


def _regex_fallback_parse(text: str) -> Dict[str, Any]:
    metadata = {
        'doi': '', 'udc': '', 'bbk': '', 'title_ru': '',
        'authors': [], 'affiliations': [], 'abstract_ru': '',
        'keywords_ru': [], 'funding': '', 'references_count': 0
    }
    
    doi_match = re.search(r'<article-id[^>]*pub-id-type="doi"[^>]*>([^<]+)</article-id>', text)
    if doi_match:
        metadata['doi'] = doi_match.group(1).strip()
    
    udc_match = re.search(r'УДК\s*([\d\.\:\-\+\(\)]+)', text, re.IGNORECASE)
    if udc_match:
        metadata['udc'] = udc_match.group(1).strip()
    
    bbk_match = re.search(r'ББК\s*([\w\.\(\)]+)', text, re.IGNORECASE)
    if bbk_match:
        metadata['bbk'] = bbk_match.group(1).strip()
    
    title_match = re.search(r'<article-title>([^<]+)</article-title>', text, re.DOTALL)
    if title_match:
        metadata['title_ru'] = title_match.group(1).strip()
    
    for contrib_match in re.finditer(r'<contrib[^>]*contrib-type="author"[^>]*>(.*?)</contrib>', text, re.DOTALL):
        contrib_text = contrib_match.group(1)
        author = {'name': '', 'orcid': ''}
        surname = re.search(r'<surname>([^<]+)</surname>', contrib_text)
        given = re.search(r'<given-names>([^<]+)</given-names>', contrib_text)
        if surname:
            name = surname.group(1).strip()
            if given:
                name += ' ' + given.group(1).strip()
            author['name'] = name
        orcid = re.search(r'<contrib-id[^>]*contrib-id-type="orcid"[^>]*>([^<]+)</contrib-id>', contrib_text)
        if orcid:
            author['orcid'] = orcid.group(1).strip()
        if author['name']:
            metadata['authors'].append(author)
    
    for aff_match in re.finditer(r'<aff[^>]*id="([^"]*)"[^>]*>(.*?)</aff>', text, re.DOTALL):
        aff_id, aff_text = aff_match.group(1), aff_match.group(2)
        aff_data = {'id': aff_id, 'institution': '', 'city': '', 'country': ''}
        inst = re.search(r'<institution>([^<]+)</institution>', aff_text)
        if inst:
            aff_data['institution'] = inst.group(1).strip()
        metadata['affiliations'].append(aff_data)
    
    abstract_ru = re.search(r'<abstract[^>]*>(.*?)</abstract>', text, re.DOTALL)
    if abstract_ru:
        text_parts = re.findall(r'<p>(.*?)</p>', abstract_ru.group(1), re.DOTALL)
        metadata['abstract_ru'] = ' '.join(t.strip() for t in text_parts)
    
    kwd_group = re.search(r'<kwd-group[^>]*>(.*?)</kwd-group>', text, re.DOTALL)
    if kwd_group:
        keywords = re.findall(r'<kwd>([^<]+)</kwd>', kwd_group.group(1))
        metadata['keywords_ru'] = [k.strip() for k in keywords if k.strip()]
    
    refs = re.findall(r'<ref[^>]*>', text)
    metadata['references_count'] = len(refs)
    
    return metadata

_GOST_METADATA_PROMPT = """Ты — эксперт-библиограф. Извлеки метаданные из научной статьи в формате JATS XML.

ТРЕБОВАНИЯ ГОСТ Р 7.0.7-2021:
1. УДК, DOI, ББК
2. Название (русский)
3. Авторы: ФИО, ORCID
4. Аффилиации: организация, город, страна
5. Аннотация (русский)
6. Ключевые слова (русский)
7. Финансирование
8. Список литературы

ФОРМАТ — строго JATS XML без markdown:

<?xml version="1.0" encoding="UTF-8"?>
<article article-type="research-article">
  <front>
    <article-meta>
      <article-id pub-id-type="doi">10.xxxx/xxxxx</article-id>
      <article-categories>
        <subj-group subj-group-type="heading">
          <subject>УДК 123.456</subject>
        </subj-group>
        <subj-group subj-group-type="bbk">
          <subject>ББК 67.89</subject>
        </subj-group>
      </article-categories>
      <title-group>
        <article-title>Название</article-title>
      </title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name>
            <surname>Фамилия</surname>
            <given-names>Имя Отчество</given-names>
          </name>
          <xref ref-type="aff" rid="aff1"/>
          <contrib-id contrib-id-type="orcid">0000-0000-0000-0000</contrib-id>
        </contrib>
      </contrib-group>
      <aff id="aff1">
        <institution>Организация</institution>
        <addr-line>Город</addr-line>
        <country>Страна</country>
      </aff>
      <abstract xml:lang="ru">
        <p>Аннотация</p>
      </abstract>
      <kwd-group xml:lang="ru">
        <kwd>слово1</kwd>
        <kwd>слово2</kwd>
      </kwd-group>
      <funding-group>
        <funding-statement>Финансирование</funding-statement>
      </funding-group>
    </article-meta>
  </front>
  <back>
    <ref-list>
      <ref><mixed-citation>Ссылка</mixed-citation></ref>
    </ref-list>
  </back>
</article>

ПРАВИЛА:
- НЕ используй markdown
- Все теги закрыты
- Если элемента нет — пропусти
- Извлекай ТОЛЬКО то, что явно есть в тексте

Текст статьи:
{}"""


def _extract_metadata_via_llm(article_text: str) -> Optional[Dict]:
    if not HAS_AI_ENGINE:
        return None
    
    text_for_llm = article_text
    prompt = _GOST_METADATA_PROMPT.format(text_for_llm)
    
    try:
        response = generate_response(prompt)

        if not response or not isinstance(response, str):
            logger.warning("[RAG LLM] LLM вернул пустой или некорректный ответ")
            return None
        
        response_clean = re.sub(r'```xml\s*', '', response)
        response_clean = re.sub(r'```\s*', '', response_clean).strip()
        
        metadata = None
        xml_text = response_clean

        secure_parser = ET.XMLParser(resolve_entities=False, no_network=True)
        
        try:
            root = ET.fromstring(response_clean.encode('utf-8'), parser=secure_parser)
            metadata = _parse_jats_xml(root)
        except ET.XMLSyntaxError:
            xml_match = re.search(r'<\?xml.*?</article>', response_clean, re.DOTALL)
            if not xml_match:
                xml_match = re.search(r'<article.*?</article>', response_clean, re.DOTALL)
            
            if xml_match:
                xml_text = xml_match.group(0)
                try:
                    root = ET.fromstring(xml_text.encode('utf-8'), parser=secure_parser)
                    metadata = _parse_jats_xml(root)
                except ET.XMLSyntaxError:
                    pass
        
        if not metadata:
            metadata = _regex_fallback_parse(response_clean)
            xml_text = response_clean
        
        if HAS_TEXTUAL_GROUNDING and VERIFY_METADATA and xml_text and metadata:
            try:
                fixed_xml, fixes = pipeline_verify_and_fix(xml_text, article_text)
                if fixes:
                    try:
                        root = ET.fromstring(fixed_xml.encode('utf-8'), parser=secure_parser)
                        metadata = _parse_jats_xml(root)
                    except Exception:
                        pass
            except Exception:
                pass
        
        return metadata if metadata and any(v for v in metadata.values() if v) else None
        
    except Exception as e:
        logger.error(f"[RAG LLM] ✗ Ошибка извлечения метаданных: {e}", exc_info=True)
        return None


def get_or_extract_metadata(article_text: str, use_cache: bool = True) -> Optional[Dict]:
    if use_cache:
        cached = _get_cached_metadata(article_text)
        if cached is not None:
            return cached
    
    if not ENABLE_LLM_METADATA or not _is_llm_available():
        return None
    
    metadata = _extract_metadata_via_llm(article_text)
    if metadata:
        _store_cached_metadata(article_text, metadata)
    
    return metadata


def _extract_and_store_metadata_for_example(example_id: str, article_text: str):
    metadata = get_or_extract_metadata(article_text)
    if not metadata:
        return
    
    db = get_rag_db()
    for ex in db["examples"]:
        if ex["id"] == example_id:
            ex["metadata"] = metadata
            ex["metadata_hash"] = _compute_article_hash(article_text)
            break
    save_rag_db(db)

def _migrate_old_examples():
    if not AUTO_MIGRATE or not ENABLE_LLM_METADATA or not _is_llm_available():
        return
    
    db = get_rag_db()
    examples_without_metadata = [
        ex for ex in db["examples"]
        if not ex.get("metadata") and ex.get("before")
    ]
    
    if not examples_without_metadata:
        return
    
    logger.info(f"[RAG Migration] Обнаружено {len(examples_without_metadata)} примеров без метаданных")
    
    for i, ex in enumerate(examples_without_metadata, 1):
        logger.info(f"[RAG Migration] [{i}/{len(examples_without_metadata)}] Извлечение метаданных...")
        try:
            metadata = get_or_extract_metadata(ex["before"])
            if metadata:
                ex["metadata"] = metadata
                ex["metadata_hash"] = _compute_article_hash(ex["before"])
                logger.info(f"[RAG Migration] ✓ Миграция успешна")
            else:
                logger.warning(f"[RAG Migration] ⚠ Не удалось извлечь метаданные")
        except Exception as e:
            logger.error(f"[RAG Migration] ✗ Ошибка: {e}", exc_info=True)
    
    save_rag_db(db)
    logger.info(f"[RAG Migration] Миграция завершена")

def extract_text_signature(text, head_lines=15):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    head = lines[:head_lines]
    
    sig = {'raw_head': head}
    if not head:
        return sig
    
    first = head[0]
    alpha_count = sum(1 for c in first if c.isalpha())
    upper_count = sum(1 for c in first if c.isupper() and c.isalpha())
    upper_ratio = upper_count / alpha_count if alpha_count > 0 else 0
    
    meta_starts = [
        r'^doi\b', r'^http', r'^\d{4}[\.\-/]',
        r'^udc\b', r'^удк\b', r'^bbk\b', r'^ббк\b',
        r'^isbn\b', r'^issn\b', r'^№', r'^#',
    ]
    first_looks_like_metadata = any(re.match(p, first, re.IGNORECASE) for p in meta_starts)
    
    first_is_section_like = (
        not first_looks_like_metadata and
        alpha_count >= 10 and
        (upper_ratio > 0.5 or len(first) > 30)
    )
    
    sig['first_line_is_metadata'] = first_looks_like_metadata
    sig['first_line_is_section_like'] = first_is_section_like
    sig['first_line_upper_ratio'] = upper_ratio
    sig['first_line_length'] = len(first)
    
    lengths = [len(l) for l in head]
    sig['avg_line_length'] = sum(lengths) / len(lengths) if lengths else 0
    sig['short_line_ratio'] = sum(1 for l in lengths if l < 30) / len(lengths)
    sig['long_line_ratio'] = sum(1 for l in lengths if l > 80) / len(lengths)
    
    head_text = '\n'.join(head)
    sig['numeric_density'] = len(re.findall(r'\d+', head_text)) / max(len(head_text), 1) * 100
    
    allcaps = sum(1 for l in head if len(l) > 8 and l.isupper() and any(c.isalpha() for c in l))
    sig['allcaps_line_ratio'] = allcaps / len(head)
    
    author_patterns = re.findall(
        r'[А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z]\.\s*[А-ЯЁA-Z]\.'
        r'|,\s*[А-ЯЁA-Z]\.\s*[А-ЯЁA-Z]?\.\s*[А-ЯЁA-Z][а-яёa-z]+)',
        head_text
    )
    sig['estimated_author_count'] = min(len(set(author_patterns)), 10)
    
    cyrillic = sum(1 for c in head_text if '\u0400' <= c <= '\u04FF')
    latin = sum(1 for c in head_text if c.isascii() and c.isalpha())
    sig['cyrillic_ratio'] = cyrillic / (cyrillic + latin) if (cyrillic + latin) > 0 else 0.5
    
    sig['has_email'] = bool('@' in text[:3000])
    sig['has_orcid_pattern'] = bool(re.search(r'\d{4}-\d{4}-\d{4}-\d{3}[\dX]', text[:3000]))
    
    return sig


def compare_signatures(sig1, sig2):
    weights = {
        'first_line_is_section_like': 3.0,
        'first_line_is_metadata': 2.0,
        'estimated_author_count': 2.5,
        'allcaps_line_ratio': 1.5,
        'short_line_ratio': 1.0,
        'long_line_ratio': 1.0,
        'numeric_density': 0.5,
        'cyrillic_ratio': 0.3,
        'has_email': 0.5,
        'has_orcid_pattern': 0.5,
    }
    
    total_weight = 0
    weighted_sum = 0
    
    for key, weight in weights.items():
        v1 = sig1.get(key)
        v2 = sig2.get(key)
        if v1 is None or v2 is None:
            continue
        
        if isinstance(v1, bool): v1 = 1.0 if v1 else 0.0
        if isinstance(v2, bool): v2 = 1.0 if v2 else 0.0
        
        if key == 'estimated_author_count':
            if isinstance(v1, int) and isinstance(v2, int):
                sim = 1.0 if v1 == v2 else (0.5 if abs(v1 - v2) == 1 else 0.0)
            else:
                sim = 1.0 - min(abs(v1 - v2) / 5.0, 1.0)
        else:
            sim = 1.0 - abs(float(v1) - float(v2))
        
        weighted_sum += sim * weight
        total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0

class _MetadataComparator:
    
    @staticmethod
    def jaccard_similarity(list1, list2):
        if not list1 or not list2:
            return 0.0
        set1, set2 = set(str(x) for x in list1), set(str(x) for x in list2)
        return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0.0
    
    @staticmethod
    def fuzzy_string_similarity(str1, str2):
        if not str1 or not str2:
            return 0.0
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    @staticmethod
    def fuzzy_list_similarity(list1, list2):
        if not list1 or not list2:
            return 0.0
        str1 = [str(x).lower() for x in list1]
        str2 = [str(x).lower() for x in list2]
        
        max_matches = 0
        for s1 in str1:
            best_match = max([SequenceMatcher(None, s1, s2).ratio() for s2 in str2])
            if best_match > 0.6:
                max_matches += 1
        for s2 in str2:
            best_match = max([SequenceMatcher(None, s2, s1).ratio() for s1 in str1])
            if best_match > 0.6:
                max_matches += 1
        
        total = len(str1) + len(str2)
        return max_matches / total if total > 0 else 0.0
    
    @staticmethod
    def compare_authors(authors1, authors2):
        if not authors1 or not authors2:
            return 0.0
        count_sim = 1.0 - abs(len(authors1) - len(authors2)) / max(len(authors1), len(authors2))
        names1 = [a.get('name', '') for a in authors1]
        names2 = [a.get('name', '') for a in authors2]
        name_sim = _MetadataComparator.jaccard_similarity(names1, names2)
        orcid1 = any(a.get('orcid') for a in authors1)
        orcid2 = any(a.get('orcid') for a in authors2)
        orcid_sim = 1.0 if orcid1 == orcid2 else 0.0
        orcid_count1 = sum(1 for a in authors1 if a.get('orcid'))
        orcid_count2 = sum(1 for a in authors2 if a.get('orcid'))
        orcid_count_sim = 1.0 - abs(orcid_count1 - orcid_count2) / max(orcid_count1, orcid_count2, 1)
        return count_sim * 0.2 + name_sim * 0.3 + orcid_sim * 0.3 + orcid_count_sim * 0.2
    
    @staticmethod
    def compare_affiliations(affs1, affs2):
        if not affs1 or not affs2:
            return 0.0
        count_sim = 1.0 - abs(len(affs1) - len(affs2)) / max(len(affs1), len(affs2))
        names1 = [a.get('institution', '') for a in affs1]
        names2 = [a.get('institution', '') for a in affs2]
        name_sim = _MetadataComparator.fuzzy_list_similarity(names1, names2)
        cities1 = [a.get('city', '') for a in affs1]
        cities2 = [a.get('city', '') for a in affs2]
        city_sim = _MetadataComparator.jaccard_similarity(cities1, cities2)
        return count_sim * 0.3 + name_sim * 0.5 + city_sim * 0.2
    
    @staticmethod
    def calculate_similarity(meta1, meta2):
        scores = {}
        scores['udc'] = _MetadataComparator.fuzzy_string_similarity(meta1.get('udc', ''), meta2.get('udc', ''))
        scores['doi'] = 1.0 if (meta1.get('doi') and meta2.get('doi')) else 0.0
        scores['title_ru'] = _MetadataComparator.fuzzy_string_similarity(meta1.get('title_ru', ''), meta2.get('title_ru', ''))
        scores['authors'] = _MetadataComparator.compare_authors(meta1.get('authors', []), meta2.get('authors', []))
        scores['affiliations'] = _MetadataComparator.compare_affiliations(meta1.get('affiliations', []), meta2.get('affiliations', []))
        scores['abstract_ru'] = _MetadataComparator.fuzzy_string_similarity(meta1.get('abstract_ru', ''), meta2.get('abstract_ru', ''))
        scores['keywords_ru'] = _MetadataComparator.fuzzy_list_similarity(meta1.get('keywords_ru', []), meta2.get('keywords_ru', []))
        kw_count1 = len(meta1.get('keywords_ru', []))
        kw_count2 = len(meta2.get('keywords_ru', []))
        scores['keywords_count'] = 1.0 - abs(kw_count1 - kw_count2) / max(kw_count1, kw_count2, 1)
        scores['bbk'] = _MetadataComparator.fuzzy_string_similarity(meta1.get('bbk', ''), meta2.get('bbk', ''))
        scores['funding'] = 1.0 if (meta1.get('funding') and meta2.get('funding')) else 0.0
        refs1 = meta1.get('references_count', 0)
        refs2 = meta2.get('references_count', 0)
        scores['references'] = 1.0 - min(abs(refs1 - refs2) / 50.0, 1.0)
        return scores
    
    @staticmethod
    def weighted_similarity(meta1, meta2, weights=None):
        if weights is None:
            weights = {
                'udc': 0.08, 'doi': 0.05, 'title_ru': 0.08,
                'authors': 0.25, 'affiliations': 0.15,
                'abstract_ru': 0.05, 'keywords_ru': 0.12,
                'keywords_count': 0.08, 'bbk': 0.05,
                'funding': 0.04, 'references': 0.05,
            }
        scores = _MetadataComparator.calculate_similarity(meta1, meta2)
        total_weight = sum(weights.values())
        weighted_sum = sum(scores.get(k, 0) * v for k, v in weights.items())
        return weighted_sum / total_weight if total_weight > 0 else 0.0

def _llm_judge_select(target_meta: Dict, candidates: List[Tuple[int, float, Dict]],
examples: List[Dict], target_article_text: str = "") -> Optional[Tuple[int, str]]:
    if not HAS_AI_ENGINE or not _is_llm_available():
        return None
        
    prompt = """Ты — эксперт-библиограф. Выбери наиболее подходящий пример (few-shot) для парсинга целевой научной статьи.
КРИТЕРИИ ВЫБОРА (по важности):
1. СТРУКТУРНОЕ СООТВЕТСТВИЕ (критично):
- Количество авторов должно СОВПАДАТЬ
- Наличие ORCID должно СОВПАДАТЬ (оба есть или оба нет)
- Количество аффилиаций должно быть похоже
- Количество ключевых слов должно быть похоже
2. БИБЛИОГРАФИЧЕСКОЕ СООТВЕТСТВИЕ:
- УДК из той же области
- ББК похож
3. КОНТЕНТНОЕ СООТВЕТСТВИЕ:
- Ключевые слова пересекаются
4. СТРУКТУРНОЕ СХОДСТВО ТЕКСТА (по превью):
- Сравни превью (первые и последние 20 строк) целевой статьи и кандидатов.
- Обращай внимание на наличие специфических блоков (например, "Финансирование", "Конфликт интересов", "Благодарность", "Список литературы", "Информация об авторах").
- Шаблон должен максимально точно повторять структуру оригинала.

ЦЕЛЕВАЯ СТАТЬЯ:
{target_info}

КАНДИДАТЫ:
{candidates_info}

ОТВЕТЬ СТРОГО В СЛЕДУЮЩЕМ ФОРМАТЕ:
ВЫБОР: <номер_кандидата>
ОБОСНОВАНИЕ: <2-3 предложения>

ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:
ВЫБОР: 2
ОБОСНОВАНИЕ: Кандидат 2 имеет совпадающее количество авторов (2), наличие ORCID и аналогичный блок "Финансирование" в конце текста, как в целевой статье.

Теперь твой ответ:"""

    def format_meta_info(meta):
        authors = len(meta.get('authors', []))
        orcid = 'Да' if any(a.get('orcid') for a in meta.get('authors', [])) else 'Нет'
        affs = len(meta.get('affiliations', []))
        udc = meta.get('udc', '—')
        bbk = meta.get('bbk', '—')
        doi = meta.get('doi', '—')
        keywords = ', '.join(meta.get('keywords_ru', [])[:5])
        kw_count = len(meta.get('keywords_ru', []))
        refs = meta.get('references_count', 0)
        return f"""Авторов: {authors}, ORCID: {orcid}
Аффилиаций: {affs}
УДК: {udc}, ББК: {bbk}
DOI: {doi}
Ключевых слов ({kw_count}): {keywords}
Ссылок: {refs}"""

    target_lines = [l for l in target_article_text.split('\n') if l.strip()]
    if len(target_lines) > 40:
        target_preview = "\n".join(target_lines[:20] + ["... (пропущено) ..."] + target_lines[-20:])
    else:
        target_preview = "\n".join(target_lines)
        
    target_info = format_meta_info(target_meta)
    if target_preview:
        target_info += f"\n\nПРЕВЬЮ ТЕКСТА (первые и последние 20 строк):\n{target_preview}"
        
    candidates_info = ""
    for rank, ((_, score, meta), ex) in enumerate(zip(candidates, examples), 1):
        before_text = ex.get("before", "")
        lines = [l for l in before_text.split('\n') if l.strip()]
        if len(lines) > 40:
            preview_lines = lines[:20] + ["... (пропущено) ..."] + lines[-20:]
        else:
            preview_lines = lines
        preview_text = "\n".join(preview_lines)
        
        candidates_info += f"\nКАНДИДАТ {rank} (score={score:.3f}):\n"
        candidates_info += format_meta_info(meta) + "\n"
        if preview_text:
            candidates_info += f"ПРЕВЬЮ ТЕКСТА (первые и последние 20 строк):\n{preview_text}\n"
            
    full_prompt = prompt.format(target_info=target_info, candidates_info=candidates_info)
    
    try:
        response = generate_response(full_prompt)
        if not response or not isinstance(response, str):
            logger.warning("[RAG Judge] ⚠ LLM вернул пустой или некорректный ответ")
            return None
        choice_rank = None
        patterns = [
            r'ВЫБОР:\s*(\d+)',
            r'Выбор:\s*(\d+)',
            r'Кандидат\s+(\d+)',
            r'Статья\s+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    rank = int(match.group(1))
                    if 1 <= rank <= len(candidates):
                        choice_rank = rank
                        break
                except ValueError:
                    continue
                    
        justification = "Не извлечено"
        for pattern in [r'ОБОСНОВАНИЕ:\s*(.+?)(?:\n|$)', r'Обоснование:\s*(.+?)(?:\n|$)']:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                justification = match.group(1).strip()
                if len(justification) > 200:
                    justification = justification[:200] + "..."
                break
                
        if choice_rank is not None:
            return choice_rank - 1, justification
        return None
        
    except Exception as e:
        logger.error(f"[RAG Judge] ✗ Ошибка: {e}", exc_info=True)
        return None

def find_similar_example(article_text, threshold=0.4, use_llm=True):
    db = get_rag_db()
    
    if not db["examples"]:
        logger.info("[RAG] База пуста")
        return None
    
    llm_available = use_llm and ENABLE_LLM_METADATA and _is_llm_available()
    
    if not llm_available:
        logger.info(f"[RAG] LLM недоступен, использую структурный метод (fallback)")
        return _find_similar_structural(article_text, threshold)
    
    _migrate_old_examples()
    
    target_meta = get_or_extract_metadata(article_text)
    if not target_meta:
        logger.warning("[RAG] Не удалось извлечь метаданные целевой статьи, использую fallback")
        return _find_similar_structural(article_text, threshold)
    
    candidates = []
    for example in db["examples"]:
        ex_meta = example.get("metadata")
        if not ex_meta:
            ex_meta = get_or_extract_metadata(example.get("before", ""))
            if ex_meta:
                example["metadata"] = ex_meta
                example["metadata_hash"] = _compute_article_hash(example.get("before", ""))
                save_rag_db(db)
        
        if ex_meta:
            score = _MetadataComparator.weighted_similarity(target_meta, ex_meta)
            candidates.append((example, score, ex_meta))
    
    if not candidates:
        logger.warning("[RAG] Нет примеров с метаданными, использую fallback")
        return _find_similar_structural(article_text, threshold)
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    top_k_candidates = candidates[:TOP_K]

    if USE_LLM_JUDGE and len(top_k_candidates) > 1:
        indexed_candidates = [(i, score, meta) for i, (ex, score, meta) in enumerate(top_k_candidates)]
        judge_result = _llm_judge_select(
            target_meta, 
            [(i, score, meta) for i, score, meta in indexed_candidates],
            [ex for ex, _, _ in top_k_candidates],
            article_text
        )
        
        if judge_result is not None:
            chosen_idx, justification = judge_result
            chosen_example, chosen_score, chosen_meta = top_k_candidates[chosen_idx]
            logger.info(f"[RAG] LLM-судья выбрал пример (score={chosen_score:.3f}, id={chosen_example['id'][:8]}...)")
            logger.info(f"[RAG]   Обоснование: {justification[:100]}...")
            method = "llm_metadata_with_judge"
        else:
            chosen_example, chosen_score, chosen_meta = top_k_candidates[0]
            logger.warning(f"[RAG] LLM-судья не сработал, использую топ-1 (score={chosen_score:.3f})")
            method = "llm_metadata_top1"
    else:
        chosen_example, chosen_score, chosen_meta = top_k_candidates[0]
        logger.info(f"[RAG] Найден пример (score={chosen_score:.3f}, id={chosen_example['id'][:8]}...)")
        method = "llm_metadata_top1"
    
    if chosen_score < threshold:
        logger.info(f"[RAG] Лучший кандидат ниже порога {threshold} (score={chosen_score:.3f})")
        return None
    
    return {
        "before": chosen_example["before"],
        "after": chosen_example["after"],
        "id": chosen_example["id"],
        "similarity": chosen_score,
        "source_template": chosen_example.get("source_template", "unknown"),
        "added": chosen_example.get("timestamp", ""),
        "metadata": chosen_meta,
        "method": method,
        "signature": extract_text_signature(chosen_example["before"]),
    }


def _find_similar_structural(article_text, threshold=0.4):
    db = get_rag_db()
    target_sig = extract_text_signature(article_text)
    
    best_match = None
    best_score = 0.0
    
    for example in db["examples"]:
        ex_sig = extract_text_signature(example["before"])
        score = compare_signatures(target_sig, ex_sig)
        
        if score > best_score:
            best_score = score
            best_match = {
                "before": example["before"],
                "after": example["after"],
                "id": example["id"],
                "similarity": score,
                "source_template": example.get("source_template", "unknown"),
                "added": example.get("timestamp", ""),
                "signature": ex_sig,
                "metadata": example.get("metadata"),
                "method": "structural_fallback",
            }
    
    if best_match and best_score >= threshold:
        sig = best_match["signature"]
        logger.info(f"[RAG] Найден пример (similarity={best_score:.3f}, id={best_match['id'][:8]}...)")
        logger.info(f"[RAG]   Признаки лучшего примера:")
        logger.info(f"[RAG]     section: {'ДА' if sig.get('first_line_is_section_like') else 'НЕТ'}")
        logger.info(f"[RAG]     авторов≈{sig.get('estimated_author_count', 0)}")
        logger.info(f"[RAG]     cyrillic={sig.get('cyrillic_ratio', 0):.2f}")
        return best_match
    else:
        logger.info(f"[RAG] Нет примеров выше порога {threshold} (лучший={best_score:.3f})")
        return None

def refresh_metadata_for_all_examples(force: bool = False):
    if not _is_llm_available():
        logger.warning("[RAG] LLM недоступен")
        return
    
    db = get_rag_db()
    refreshed = 0
    
    for i, ex in enumerate(db["examples"], 1):
        if not force and ex.get("metadata"):
            logger.info(f"[RAG] [{i}/{len(db['examples'])}] Пропуск (уже есть)")
            continue
        
        logger.info(f"[RAG] [{i}/{len(db['examples'])}] Извлечение метаданных...")
        try:
            metadata = get_or_extract_metadata(ex.get("before", ""), use_cache=False)
            if metadata:
                ex["metadata"] = metadata
                ex["metadata_hash"] = _compute_article_hash(ex.get("before", ""))
                refreshed += 1
                logger.info(f"[RAG] Успешно")
            else:
                logger.warning(f"[RAG] Не удалось")
        except Exception as e:
            logger.error(f"[RAG] Ошибка: {e}", exc_info=True)
    
    save_rag_db(db)
    logger.info(f"[RAG] Обновлено {refreshed} примеров")


def get_example_metadata(example_id: str) -> Optional[Dict]:
    db = get_rag_db()
    for ex in db["examples"]:
        if ex["id"] == example_id:
            if ex.get("metadata"):
                return ex["metadata"]
            metadata = get_or_extract_metadata(ex.get("before", ""))
            if metadata:
                ex["metadata"] = metadata
                save_rag_db(db)
            return metadata
    return None
