# Template in Flask:

## Template là gì?

Templace là các file chứa các dữ liệu tĩnh là các placeholder cho các dữ liệu động. Một templace được render với các dữ liệu cụ thể để sinh ra tệp cuối cùng.

Flask sử dụng thư viện Jinja để render template

Ta sẽ sử dụng template để render các thành phần HTML hiển thị trên user browser.

**Một số dấu phân cách trong Jinja**: Dấu phân cách sử dụng để ngăn cách các thành phần tĩnh với các biểu thức cú pháp Jinja

- Sử dụng cặp dấu `{{` và `}}` để biểu thị biểu thức được hiển thị như output.
- Sử dụng cặp dấu `{%` và `%}` để biểu thị các lệnh điều khiển như `for` hoặc `if`

Trong chương trình, ta sẽ phân trang HTML thành các layout riêng biệt để dễ quản lý và tái sử dụng.

## Base layout
