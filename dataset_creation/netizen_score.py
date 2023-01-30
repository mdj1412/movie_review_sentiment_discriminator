import math
from bs4 import BeautifulSoup
from tqdm import trange
import urllib.request


review_url_form = 'https://movie.naver.com/movie/point/af/list.naver?&page={}'

# Main Function
def get_review_data(start_page=1, end_page=-1):
    url = review_url_form.format(start_page)

    # 해당 영화의 마지막 리뷰 페이지 가져와서 내가 전달한 인자와 비교
    max_page = calc_max_page(url, end_page)

    if max_page == False:
        return "Failed: please check end_page"

    comments = []
    for page in trange(start_page, max_page + 1):
        url = review_url_form.format(page)

        current_page_comments = get_a_page(url)
        comments += current_page_comments

    return comments




def get_a_page(url):
    # html 가져오기
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')

    comments = []
    # 한 페이지에 있는 score, text, user_name 가져오기 ( 각각 10개씩 )
    for row in soup.select('table[class=list_netizen] tbody tr'):#각각의 리뷰들
        try:
            # from IPython import embed; embed()

            score = get_score(row)
            text = get_text(row)
            title = get_title(row)

            # 가끔 "관람객" 표시되어 있는 것이 있음 => 제거
            text = remove_formal_text(text)
            # Transform Data(score, text, user_id) to Dictionary
            comments.append(return_comment_form(score, text, title))
        except Exception as e:
            print("Get Problem !")
            return e

    return comments









# 가끔 "관람객" 표시되어 있는 것이 있음 => 제거
def remove_formal_text(text):
    if text[:4] == '관람객\n':
        return text[4:].strip()
    return text


def get_score(row):
    score = int(row.select('div[class=list_netizen_score] em')[0].text.strip())
    return score


def get_text(row):
    abc = row.select('td[class=title]')
    aaa = [a.strip() for a in abc[0].text.split('\n') if a.strip() != '']
    text = aaa[-2]
    return text

def get_title(row):
    # text = row.select('a[class=movie color_b]')
    # text = row.select('a')
    text = row.select('a.movie.color_b')
    return text[0].text


def return_comment_form(score, text, title):
    comment = {'score': score,
               'text': text,
               'title' : title
               }
    return comment











# 해당 영화의 마지막 리뷰 페이지 가져와서 내가 전달한 인자와 비교
def calc_max_page(url, end_page):
    max_page = min(end_page, get_max_page(url))
    return max_page if max_page > 0 else False



# 해당 영화의 마지막 리뷰 페이지 가져오기
def get_max_page(url):
    # html 정보 가져오기
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')

    try:
        # 관람객 평점 건수를 가져와서 1,000 -> 1000 으로 바꾸기
        num_comments = int(soup.select('strong[class="c_88 fs_11"]')[-1].text.replace(',', ''))
        # 한 페이지에 들어갈 수 있는 최대 평점 수 => 10개
        return math.ceil(num_comments / 10)
    except Exception as e:
        return -1