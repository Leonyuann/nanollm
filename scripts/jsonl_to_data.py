from pathlib import Path
import argparse
import json
import shutil
import tempfile


from data_process.filter import gopher_quality_filter, mask_emails, mask_phone_numbers, mask_ips
from data_process.deduplication import exact_deduplication_on_files, minhash_deduplicatin
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, default="data/cc_sample.jsonl")
    parser.add_argument("--target_file", type=str, default="data/cc_sample.txt")
    parser.add_argument("--filter_log_path", type=str, default="outputs/logs/filter_log.json")
    parser.add_argument("--dedup_log_path", type=str, default="outputs/logs/dedup_log.json")

    return parser.parse_args()

def main(args):
    target_file: Path = Path(args.target_file)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="jsonl_to_data_",
        dir=Path(args.input_file).parent,
    ) as tmp_dir:
    # Filter based on Gopher rules and mask personal information
        filter_log = __filter_and_dump_jsonl(args.input_file, tmp_dir)

        filter_log_path = Path(args.filter_log_path)
        filter_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(filter_log_path,mode="w",encoding="utf-8") as f:
            json.dump(filter_log, f, indent=4, ensure_ascii=False)

        dedup_log = __deduplicate(tmp_dir)
        dedup_log_path = Path(args.dedup_log_path)
        dedup_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dedup_log_path,mode="w",encoding="utf-8") as f:
            json.dump(dedup_log, f, indent=4, ensure_ascii=False)

        
        tmp_dir = Path(tmp_dir)
        with open(target_file, mode="w", encoding="utf-8") as f:
            for input_file in sorted(tmp_dir.iterdir()):
                if input_file.is_file() is False:
                    continue
                
                f.write(input_file.read_text())
                f.write("\n<|endoftext|>\n")
    
    return

def __filter_and_dump_jsonl(source_file: str, target_dir: str = "data/tmp") -> dict:
    """Filter record in .jsonl and dump them to specific directory
    """
    filter_log = {}
    target_dir : Path= Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Iterate on each record
    with open(source_file,mode="r",encoding="utf-8") as f:
        for index, json_record in enumerate(f, start=1):
            record = json.loads(json_record)
            filter_log["total_text"] = filter_log.get("total_text",0) + 1
            filter_log["total_ch"] = filter_log.get("total_ch", 0) + len(record["text"])

            if record["language"] in {"eng", "en", "english", "__label__en"}:
                record["language"] = "english"
            # If the text doesn't pass Gopher rules
            if gopher_quality_filter(record["text"],record["language"]) == False:
                filter_log["filter_text"] = filter_log.get("filter_text", 0) + 1
                filter_log["filter_ch"] = filter_log.get("filter_ch", 0) + len(record["text"])
                continue

            # If passed, mask personal information of text
            record["text"], emails = mask_emails(record["text"])
            record["text"], phone_number = mask_phone_numbers(record["text"])
            record["text"], ips  = mask_ips(record["text"])

            filter_log["filter_emails"] = filter_log.get("filter_emails", 0) + emails
            filter_log["filter_phone_number"] = filter_log.get("filter_phone_number", 0) + phone_number
            filter_log["filter_ips"] = filter_log.get("filter_ips", 0) + ips

            # Write to target file
            target_file: Path = target_dir / f"{index}.txt"
            target_file.write_text(record["text"])

    return filter_log

def __deduplicate(
    source_dir: str,
    num_has_fun: int = 128,
    num_bands: int = 16,
    ngram: int = 5,
    threshold: float = 0.85,
) -> dict:
    source_dir = Path(source_dir)

    with tempfile.TemporaryDirectory(
        prefix=f"{source_dir.name}_exact_dedup_",
        dir=source_dir.parent,
    ) as exact_tmp_dir:
        exact_tmp_dir = Path(exact_tmp_dir)

        input_files = [
            str(input_file)
            for input_file in source_dir.iterdir()
            if input_file.is_file()
        ]
        deleted_ch = exact_deduplication_on_files(input_files, exact_tmp_dir)

        shutil.rmtree(source_dir)
        exact_tmp_dir.rename(source_dir)

    with tempfile.TemporaryDirectory(
        prefix=f"{source_dir.name}_minhash_dedup_",
        dir=source_dir.parent,
    ) as minhash_tmp_dir:
        minhash_tmp_dir = Path(minhash_tmp_dir)

        input_files = [
            str(input_file)
            for input_file in source_dir.iterdir()
            if input_file.is_file()
        ]
        deleted_files = minhash_deduplicatin(
            input_files,
            output_dir=minhash_tmp_dir,
            num_hash_fun=num_has_fun,
            num_bands=num_bands,
            n_gram_length=ngram,
            theshold=threshold,
        )

        shutil.rmtree(source_dir)
        minhash_tmp_dir.rename(source_dir)

    return {"deleted_characters": deleted_ch, "deleted_files": deleted_files}

if __name__ == "__main__":
    main(parse_args())  