import math
from bs4 import BeautifulSoup
from tqdm import trange
import urllib.request

# 최신 리뷰
review_url_form = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code={}&order=newest&page={}&onlySpoilerPointYn={}'


# 해당 영화의 마지막 리뷰 페이지 가져와서 내가 전달한 인자와 비교
def calc_max_page(url, end_page):
    max_page = min(end_page, get_max_page(url))
    return max_page if max_page > 0 else False




# Main Function
def get_review_data(movie_id, start_page=1, end_page=-1, spoiler='N'):
    url = review_url_form.format(movie_id, start_page, spoiler)

    # 해당 영화의 마지막 리뷰 페이지 가져와서 내가 전달한 인자와 비교
    max_page = calc_max_page(url, end_page)

    if max_page == False:
        return "Failed: please check end_page"

    comments = []
    for page in trange(start_page, max_page + 1):
        url = review_url_form.format(movie_id, page, spoiler)
        current_page_comments = get_a_page(url)
        comments += current_page_comments
    return comments








# 가끔 "관람객" 표시되어 있는 것이 있음 => 제거
def remove_formal_text(text):
    if text[:4] == '관람객\n':
        return text[4:].strip()
    return text


def get_score(row):
    score = int(row.select('div[class=star_score] em')[0].text.strip())
    return score


def get_text(row):
    text = row.select('div[class=score_reple] p')[0].text.strip()
    return text




def return_comment_form(score, text):
    comment = {'score': score,
               'text': text
               }
    return comment



def get_a_page(url):
    # html 가져오기
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')

    comments = []
    # 한 페이지에 있는 score, text, user_name 가져오기 ( 각각 10개씩 )
    for row in soup.select('div[class=score_result] li'):#각각의 리뷰들
        try:
            score = get_score(row)
            text = get_text(row)

            # 가끔 "관람객" 표시되어 있는 것이 있음 => 제거
            text = remove_formal_text(text)
            
            # Transform Data(score, text) to Dictionary
            comments.append(return_comment_form(score, text))
        except Exception as e:
            return e
            continue

    return comments
















# 해당 영화의 마지막 리뷰 페이지 가져오기
def get_max_page(url):
    # html 정보 가져오기
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')

    try:
        # 관람객 평점 건수를 가져와서 1,000 -> 1000 으로 바꾸기
        num_comments = int(soup.select('div[class="score_total"] em')[-1].text.replace(',', ''))
        # 한 페이지에 들어갈 수 있는 최대 평점 수 => 10개
        return math.ceil(num_comments / 10)
    except Exception as e:
        return -1
