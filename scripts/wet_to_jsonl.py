from pathlib import Path
from tqdm import tqdm
import json
import argparse
from warcio.archiveiterator import ArchiveIterator

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--WET_path", type=str, default="data/cc_sample.warc.wet.gz")
    parser.add_argument("--jsonl_path", type=str, default="data/cc_sample.jsonl")
    
    return parser.parse_args()

def main(args):
    source_file = Path(args.WET_path)
    target_file = Path(args.jsonl_path)

    # Create target file
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if target_file.exists() is False:
        target_file.touch()

    with open(source_file, mode="rb") as source, open(target_file, mode="w", encoding="utf-8") as target:
        for record in tqdm(ArchiveIterator(source),desc="Processing"):
            if record.rec_type != "conversion":
                continue
            
            url = record.rec_headers.get_header("WARC-Target-URI")
            date = record.rec_headers.get_header("WARC-Date")
            record_id = record.rec_headers.get_header("WARC-Record-ID")
            language = record.rec_headers.get_header("WARC-Identified-Content-Language")
            content_type = record.rec_headers.get_header("Content-Type")
            length = record.rec_headers.get_header("Content-Length")
            text = record.content_stream().read().decode("utf-8", errors="replace")

            data = {
                "url": url,
                "data": date,
                "record_id": record_id,
                "language": language,
                "content_type": content_type,
                "length": length,
                "text": text
            }

            json.dump(data, target, ensure_ascii=False)
            target.write("\n")
            
    return       
    

if __name__ == "__main__":
    main(parse_args())