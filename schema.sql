/** Mô tả: CSDL bao gồm 2 bảng user và post. 
Bảng user cho phép lưu trữ tài khoản user (id, tên đăng nhập, mk)
Bảng post lưu trữ các bài đăng của các tài khoản ở trên
*/
DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS post;

CREATE TABLE user(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);

CREATE TABLE post(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id INTEGER NOT NULL,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    FOREIGN KEY (author_id) REFERENCES user(id)
)