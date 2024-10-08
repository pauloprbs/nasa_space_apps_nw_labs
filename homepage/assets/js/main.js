/**
* Template Name: Gp
* Template URL: https://bootstrapmade.com/gp-free-multipurpose-html-bootstrap-template/
* Updated: Aug 15 2024 with Bootstrap v5.3.3
* Author: BootstrapMade.com
* License: https://bootstrapmade.com/license/
*/
var planetsel = "earth";
var planetpath = "assets/img/planets/earth.jpeg";
const merkurybtn = document.getElementById("btn_merkury");
const venusbtn = document.getElementById("btn_venus");
const earthbtn = document.getElementById("btn_earth");
const moonbtn = document.getElementById("btn_moon");
const marsbtn = document.getElementById("btn_mars");
const planetobj = document.getElementById("planetrotate");
planetobj.style.backgroundPositionX = "0%";
var planetxoffset = 0;
var planetxcoffset = 0;

var mapsel = "earth";
var mappath = "assets/img/planets/earth.jpeg";
const earthbtn_f = document.getElementById("btn_earth_f");
const moonbtn_f = document.getElementById("btn_moon_f");
const marsbtn_f = document.getElementById("btn_mars_f");
const mapobj = document.getElementById("planetflat");

(function() {
  "use strict";

  /**
   * Apply .scrolled class to the body as the page is scrolled down
   */
  function toggleScrolled() {
    const selectBody = document.querySelector('body');
    const selectHeader = document.querySelector('#header');
    if (!selectHeader.classList.contains('scroll-up-sticky') && !selectHeader.classList.contains('sticky-top') && !selectHeader.classList.contains('fixed-top')) return;
    window.scrollY > 100 ? selectBody.classList.add('scrolled') : selectBody.classList.remove('scrolled');
  }

  document.addEventListener('scroll', toggleScrolled);
  window.addEventListener('load', toggleScrolled);

  /**
   * Mobile nav toggle
   */
  const mobileNavToggleBtn = document.querySelector('.mobile-nav-toggle');

  function mobileNavToogle() {
    document.querySelector('body').classList.toggle('mobile-nav-active');
    mobileNavToggleBtn.classList.toggle('bi-list');
    mobileNavToggleBtn.classList.toggle('bi-x');
  }
  if (mobileNavToggleBtn) {
    mobileNavToggleBtn.addEventListener('click', mobileNavToogle);
  }

  /**
   * Hide mobile nav on same-page/hash links
   */
  document.querySelectorAll('#navmenu a').forEach(navmenu => {
    navmenu.addEventListener('click', () => {
      if (document.querySelector('.mobile-nav-active')) {
        mobileNavToogle();
      }
    });

  });

  /**
   * Toggle mobile nav dropdowns
   */
  document.querySelectorAll('.navmenu .toggle-dropdown').forEach(navmenu => {
    navmenu.addEventListener('click', function(e) {
      e.preventDefault();
      this.parentNode.classList.toggle('active');
      this.parentNode.nextElementSibling.classList.toggle('dropdown-active');
      e.stopImmediatePropagation();
    });
  });

  /**
   * Preloader
   */
  const preloader = document.querySelector('#preloader');
  if (preloader) {
    window.addEventListener('load', () => {
      preloader.remove();
    });
  }

  /**
   * Scroll top button
   */
  let scrollTop = document.querySelector('.scroll-top');

  function toggleScrollTop() {
    if (scrollTop) {
      window.scrollY > 100 ? scrollTop.classList.add('active') : scrollTop.classList.remove('active');
    }
  }
  scrollTop.addEventListener('click', (e) => {
    e.preventDefault();
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });

  window.addEventListener('load', toggleScrollTop);
  document.addEventListener('scroll', toggleScrollTop);

  /**
   * Animation on scroll function and init
   */
  function aosInit() {
    AOS.init({
      duration: 600,
      easing: 'ease-in-out',
      once: true,
      mirror: false
    });
  }
  window.addEventListener('load', aosInit);

  /**
   * Init swiper sliders
   */
  function initSwiper() {
    document.querySelectorAll(".init-swiper").forEach(function(swiperElement) {
      let config = JSON.parse(
        swiperElement.querySelector(".swiper-config").innerHTML.trim()
      );

      if (swiperElement.classList.contains("swiper-tab")) {
        initSwiperWithCustomPagination(swiperElement, config);
      } else {
        new Swiper(swiperElement, config);
      }
    });
  }

  window.addEventListener("load", initSwiper);

  /**
   * Initiate glightbox
   */
  const glightbox = GLightbox({
    selector: '.glightbox'
  });

  /**
   * Init isotope layout and filters
   */
  document.querySelectorAll('.isotope-layout').forEach(function(isotopeItem) {
    let layout = isotopeItem.getAttribute('data-layout') ?? 'masonry';
    let filter = isotopeItem.getAttribute('data-default-filter') ?? '*';
    let sort = isotopeItem.getAttribute('data-sort') ?? 'original-order';

    let initIsotope;
    imagesLoaded(isotopeItem.querySelector('.isotope-container'), function() {
      initIsotope = new Isotope(isotopeItem.querySelector('.isotope-container'), {
        itemSelector: '.isotope-item',
        layoutMode: layout,
        filter: filter,
        sortBy: sort
      });
    });

    isotopeItem.querySelectorAll('.isotope-filters li').forEach(function(filters) {
      filters.addEventListener('click', function() {
        isotopeItem.querySelector('.isotope-filters .filter-active').classList.remove('filter-active');
        this.classList.add('filter-active');
        initIsotope.arrange({
          filter: this.getAttribute('data-filter')
        });
        if (typeof aosInit === 'function') {
          aosInit();
        }
      }, false);
    });

  });

  /**
   * Initiate Pure Counter
   */
  new PureCounter();

  /**
   * Correct scrolling position upon page load for URLs containing hash links.
   */
  window.addEventListener('load', function(e) {
    if (window.location.hash) {
      if (document.querySelector(window.location.hash)) {
        setTimeout(() => {
          let section = document.querySelector(window.location.hash);
          let scrollMarginTop = getComputedStyle(section).scrollMarginTop;
          window.scrollTo({
            top: section.offsetTop - parseInt(scrollMarginTop),
            behavior: 'smooth'
          });
        }, 100);
      }
    }
  });

  /**
   * Navmenu Scrollspy
   */
  let navmenulinks = document.querySelectorAll('.navmenu a');

  function navmenuScrollspy() {
    navmenulinks.forEach(navmenulink => {
      if (!navmenulink.hash) return;
      let section = document.querySelector(navmenulink.hash);
      if (!section) return;
      let position = window.scrollY + 200;
      if (position >= section.offsetTop && position <= (section.offsetTop + section.offsetHeight)) {
        document.querySelectorAll('.navmenu a.active').forEach(link => link.classList.remove('active'));
        navmenulink.classList.add('active');
      } else {
        navmenulink.classList.remove('active');
      }
    })
  }
  window.addEventListener('load', navmenuScrollspy);
  document.addEventListener('scroll', navmenuScrollspy);

  // Drop-down planet selector events
  //merkurybtn.addEventListener("click", function(){changeRotatePlanetSel("merkury", "assets/img/planets/merkury.jpg");});
  //venusbtn.addEventListener("click", function(){changeRotatePlanetSel("venus", "assets/img/planets/Venus.jpg");});
  earthbtn.addEventListener("click", function(){changeRotatePlanetSel("earth", "assets/img/planets/earth.jpeg");});
  moonbtn.addEventListener("click", function(){changeRotatePlanetSel("moon", "assets/img/planets/moon.jpg");});
  marsbtn.addEventListener("click", function(){changeRotatePlanetSel("mars", "assets/img/planets/mars.jpg");});

  earthbtn_f.addEventListener("click", function(){changeFlatPlanetSel("earth", "assets/img/planets/earth.jpeg");});
  moonbtn_f.addEventListener("click", function(){changeFlatPlanetSel("moon", "assets/img/planets/moon.jpg");});
  marsbtn_f.addEventListener("click", function(){changeFlatPlanetSel("mars", "assets/img/planets/mars.jpg");});
})();

// Planet changed
function changeRotatePlanetSel(pname, ppath){
  planetsel = pname;
  planetpath = ppath;
  planetobj.style.backgroundImage = "url(" + planetpath + ")";
  planetobj.style.backgroundPositionX = "0%";
}

function changeFlatPlanetSel(pname, ppath){
  mapsel = pname;
  mappath = ppath;
  mapobj.src = mappath;
}

/* PLANET DRAGGING START */
function planetDragStart(e){
  console.log("call!");
  e = e || window.event;
  e.preventDefault();
  planetxcoffset = e.clientX;
  planetobj.style.cursor = "grabbing";
  document.onmouseup = planetDragEnd;
  document.onpointerup = planetDragEnd;
  // call a function whenever the cursor moves:
  document.onmousemove = planetDrag;
  document.onpointermove = planetDrag;
}

function planetDrag(e){
  console.log("Drag Update!");
  e = e || window.event;
  e.preventDefault();

  planetxoffset = planetxcoffset - e.clientX;
  planetxcoffset = e.clientX;

  let aux = parseInt(planetobj.style.backgroundPositionX);
  planetobj.style.backgroundPositionX = (aux - planetxoffset) + "%";
}

function planetDragEnd(){
  console.log("Drag End!");
  planetobj.style.cursor = "grab";
  document.onmouseup = null;
  document.onmousemove = null;
  document.onpointerup = null;
  document.onpointermove = null;
}

planetobj.onmousedown = planetDragStart;
planetobj.onpointerdown = planetDragStart;
/* PLANET DRAGGING END */

/* CHATBOT START */
/* CHATBOT END */

// Easter-eggs and other removed stuff
function giveMeMercury(){
  changeRotatePlanetSel("merkury", "assets/img/planets/merkury.jpg");
}
function giveMeVenus(){
  changeRotatePlanetSel("venus", "assets/img/planets/Venus.jpg")
}