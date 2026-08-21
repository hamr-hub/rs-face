//! 一次性调试程序:用当前实现 put 一份小对象到 rustfs,看哪里挂了。
use std::time::SystemTime;
fn main() {
    let ep = std::env::var("S3_ENDPOINT").unwrap_or_else(|_| "http://rustfs:9000".into());
    let ak = std::env::var("S3_ACCESS_KEY").unwrap_or_else(|_| "rsface".into());
    let sk = std::env::var("S3_SECRET_KEY").unwrap_or_else(|_| "rsface-secret".into());
    let bucket = std::env::var("S3_BUCKET").unwrap_or_else(|_| "rsface".into());
    println!("[s3test] endpoint={ep} bucket={bucket}");
    let c = platform_lib::s3::S3Client::new(ep, "us-east-1".into(), ak, sk, bucket);
    let body = format!("hello from s3test at {:?}", SystemTime::now()).into_bytes();
    match c.put_object("s3test/hello.txt", "text/plain", body) {
        Ok(_) => println!("[s3test] PUT OK"),
        Err(e) => println!("[s3test] PUT ERR: {e}"),
    }
    match c.ensure_bucket() {
        Ok(_) => println!("[s3test] HEAD bucket OK"),
        Err(e) => println!("[s3test] HEAD bucket ERR: {e}"),
    }
}
