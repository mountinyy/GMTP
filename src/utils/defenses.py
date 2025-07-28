def get_poison_cnt(doc_ids, poison_info, p_corpus, debug=False):
    cnt = 0
    if len(p_corpus) > 0:
        for doc_id in doc_ids:
            if "poison" not in doc_id:
                continue
            corpus = p_corpus[doc_id]
            for poison_doc in poison_info["poisoned_docs"]:
                if corpus == poison_doc:
                    cnt += 1
                    break
    return cnt
