"use strict";
(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([[544], {
    5215: function(e, t, i) {
        i.d(t, {
            Z: function() {
                return wn
            }
        });
        var s = i(8908)
          , n = i.n(s)
          , r = i(9144)
          , a = i.n(r)
          , o = i(7537)
          , l = i.n(o)
          , h = i(5974)
          , d = i.n(h)
          , u = i(9603)
          , c = i.n(u)
          , p = i(3407)
          , m = i.n(p)
          , g = i(7462)
          , f = i(779)
          , _ = i(4329)
          , y = i(2260)
          , v = i(8485)
          , T = i(562)
          , b = i(7627)
          , S = i(4221)
          , k = i.n(S)
          , C = i(8925)
          , E = i(7530)
          , w = i(1489)
          , x = "8.10.0";
        const I = {}
          , P = function(e, t) {
            return I[e] = I[e] || [],
            t && (I[e] = I[e].concat(t)),
            I[e]
        }
          , A = function(e, t) {
            const i = P(e).indexOf(t);
            return !(i <= -1) && (I[e] = I[e].slice(),
            I[e].splice(i, 1),
            !0)
        }
          , L = {
            prefixed: !0
        }
          , D = [["requestFullscreen", "exitFullscreen", "fullscreenElement", "fullscreenEnabled", "fullscreenchange", "fullscreenerror", "fullscreen"], ["webkitRequestFullscreen", "webkitExitFullscreen", "webkitFullscreenElement", "webkitFullscreenEnabled", "webkitfullscreenchange", "webkitfullscreenerror", "-webkit-full-screen"]]
          , O = D[0];
        let M;
        for (let tl = 0; tl < D.length; tl++)
            if (D[tl][1]in a()) {
                M = D[tl];
                break
            }
        if (M) {
            for (let e = 0; e < M.length; e++)
                L[O[e]] = M[e];
            L.prefixed = M[0] !== O[0]
        }
        let R = [];
        const U = function e(t, i=":", s="") {
            let r, a = "info";
            const o = function(...e) {
                r("log", a, e)
            };
            return r = ((e,t,i)=>(s,r,a)=>{
                const o = t.levels[r]
                  , l = new RegExp(`^(${o})$`);
                let h = e;
                if ("log" !== s && a.unshift(s.toUpperCase() + ":"),
                i && (h = `%c${e}`,
                a.unshift(i)),
                a.unshift(h + ":"),
                R) {
                    R.push([].concat(a));
                    const e = R.length - 1e3;
                    R.splice(0, e > 0 ? e : 0)
                }
                if (!n().console)
                    return;
                let d = n().console[s];
                d || "debug" !== s || (d = n().console.info || n().console.log),
                d && o && l.test(s) && d[Array.isArray(a) ? "apply" : "call"](n().console, a)
            }
            )(t, o, s),
            o.createLogger = (n,r,a)=>{
                const o = void 0 !== r ? r : i;
                return e(`${t} ${o} ${n}`, o, void 0 !== a ? a : s)
            }
            ,
            o.createNewLogger = (t,i,s)=>e(t, i, s),
            o.levels = {
                all: "debug|log|warn|error",
                off: "",
                debug: "debug|log|warn|error",
                info: "log|warn|error",
                warn: "warn|error",
                error: "error",
                DEFAULT: a
            },
            o.level = e=>{
                if ("string" === typeof e) {
                    if (!o.levels.hasOwnProperty(e))
                        throw new Error(`"${e}" in not a valid log level`);
                    a = e
                }
                return a
            }
            ,
            (o.history = ()=>R ? [].concat(R) : []).filter = e=>(R || []).filter((t=>new RegExp(`.*${e}.*`).test(t[0]))),
            o.history.clear = ()=>{
                R && (R.length = 0)
            }
            ,
            o.history.disable = ()=>{
                null !== R && (R.length = 0,
                R = null)
            }
            ,
            o.history.enable = ()=>{
                null === R && (R = [])
            }
            ,
            o.error = (...e)=>r("error", a, e),
            o.warn = (...e)=>r("warn", a, e),
            o.debug = (...e)=>r("debug", a, e),
            o
        }("VIDEOJS")
          , B = U.createLogger
          , N = Object.prototype.toString
          , F = function(e) {
            return q(e) ? Object.keys(e) : []
        };
        function j(e, t) {
            F(e).forEach((i=>t(e[i], i)))
        }
        function $(e, t, i=0) {
            return F(e).reduce(((i,s)=>t(i, e[s], s)), i)
        }
        function q(e) {
            return !!e && "object" === typeof e
        }
        function H(e) {
            return q(e) && "[object Object]" === N.call(e) && e.constructor === Object
        }
        function V(...e) {
            const t = {};
            return e.forEach((e=>{
                e && j(e, ((e,i)=>{
                    H(e) ? (H(t[i]) || (t[i] = {}),
                    t[i] = V(t[i], e)) : t[i] = e
                }
                ))
            }
            )),
            t
        }
        function z(e={}) {
            const t = [];
            for (const i in e)
                if (e.hasOwnProperty(i)) {
                    const s = e[i];
                    t.push(s)
                }
            return t
        }
        function W(e, t, i, s=!0) {
            const n = i=>Object.defineProperty(e, t, {
                value: i,
                enumerable: !0,
                writable: !0
            })
              , r = {
                configurable: !0,
                enumerable: !0,
                get() {
                    const e = i();
                    return n(e),
                    e
                }
            };
            return s && (r.set = n),
            Object.defineProperty(e, t, r)
        }
        var G = Object.freeze({
            __proto__: null,
            each: j,
            reduce: $,
            isObject: q,
            isPlain: H,
            merge: V,
            values: z,
            defineLazyProperty: W
        });
        let K, Q = !1, X = null, Y = !1, J = !1, Z = !1, ee = !1, te = !1, ie = null, se = null, ne = null, re = !1, ae = !1, oe = !1, le = !1;
        const he = Boolean(ge() && ("ontouchstart"in n() || n().navigator.maxTouchPoints || n().DocumentTouch && n().document instanceof n().DocumentTouch))
          , de = n().navigator && n().navigator.userAgentData;
        if (de && de.platform && de.brands && (Y = "Android" === de.platform,
        Z = Boolean(de.brands.find((e=>"Microsoft Edge" === e.brand))),
        ee = Boolean(de.brands.find((e=>"Chromium" === e.brand))),
        te = !Z && ee,
        ie = se = (de.brands.find((e=>"Chromium" === e.brand)) || {}).version || null,
        ae = "Windows" === de.platform),
        !ee) {
            const e = n().navigator && n().navigator.userAgent || "";
            Q = /iPod/i.test(e),
            X = function() {
                const t = e.match(/OS (\d+)_/i);
                return t && t[1] ? t[1] : null
            }(),
            Y = /Android/i.test(e),
            K = function() {
                const t = e.match(/Android (\d+)(?:\.(\d+))?(?:\.(\d+))*/i);
                if (!t)
                    return null;
                const i = t[1] && parseFloat(t[1])
                  , s = t[2] && parseFloat(t[2]);
                return i && s ? parseFloat(t[1] + "." + t[2]) : i || null
            }(),
            J = /Firefox/i.test(e),
            Z = /Edg/i.test(e),
            ee = /Chrome/i.test(e) || /CriOS/i.test(e),
            te = !Z && ee,
            ie = se = function() {
                const t = e.match(/(Chrome|CriOS)\/(\d+)/);
                return t && t[2] ? parseFloat(t[2]) : null
            }(),
            ne = function() {
                const t = /MSIE\s(\d+)\.\d/.exec(e);
                let i = t && parseFloat(t[1]);
                return !i && /Trident\/7.0/i.test(e) && /rv:11.0/.test(e) && (i = 11),
                i
            }(),
            re = /Safari/i.test(e) && !te && !Y && !Z,
            ae = /Windows/i.test(e),
            oe = /iPad/i.test(e) || re && he && !/iPhone/i.test(e),
            le = /iPhone/i.test(e) && !oe
        }
        const ue = le || oe || Q
          , ce = (re || ue) && !te;
        var pe = Object.freeze({
            __proto__: null,
            get IS_IPOD() {
                return Q
            },
            get IOS_VERSION() {
                return X
            },
            get IS_ANDROID() {
                return Y
            },
            get ANDROID_VERSION() {
                return K
            },
            get IS_FIREFOX() {
                return J
            },
            get IS_EDGE() {
                return Z
            },
            get IS_CHROMIUM() {
                return ee
            },
            get IS_CHROME() {
                return te
            },
            get CHROMIUM_VERSION() {
                return ie
            },
            get CHROME_VERSION() {
                return se
            },
            get IE_VERSION() {
                return ne
            },
            get IS_SAFARI() {
                return re
            },
            get IS_WINDOWS() {
                return ae
            },
            get IS_IPAD() {
                return oe
            },
            get IS_IPHONE() {
                return le
            },
            TOUCH_ENABLED: he,
            IS_IOS: ue,
            IS_ANY_SAFARI: ce
        });
        function me(e) {
            return "string" === typeof e && Boolean(e.trim())
        }
        function ge() {
            return a() === n().document
        }
        function fe(e) {
            return q(e) && 1 === e.nodeType
        }
        function _e() {
            try {
                return n().parent !== n().self
            } catch (e) {
                return !0
            }
        }
        function ye(e) {
            return function(t, i) {
                if (!me(t))
                    return a()[e](null);
                me(i) && (i = a().querySelector(i));
                const s = fe(i) ? i : a();
                return s[e] && s[e](t)
            }
        }
        function ve(e="div", t={}, i={}, s) {
            const n = a().createElement(e);
            return Object.getOwnPropertyNames(t).forEach((function(e) {
                const i = t[e];
                "textContent" === e ? Te(n, i) : n[e] === i && "tabIndex" !== e || (n[e] = i)
            }
            )),
            Object.getOwnPropertyNames(i).forEach((function(e) {
                n.setAttribute(e, i[e])
            }
            )),
            s && Fe(n, s),
            n
        }
        function Te(e, t) {
            return "undefined" === typeof e.textContent ? e.innerText = t : e.textContent = t,
            e
        }
        function be(e, t) {
            t.firstChild ? t.insertBefore(e, t.firstChild) : t.appendChild(e)
        }
        function Se(e, t) {
            return function(e) {
                if (e.indexOf(" ") >= 0)
                    throw new Error("class has illegal whitespace characters")
            }(t),
            e.classList.contains(t)
        }
        function ke(e, ...t) {
            return e.classList.add(...t.reduce(((e,t)=>e.concat(t.split(/\s+/))), [])),
            e
        }
        function Ce(e, ...t) {
            return e ? (e.classList.remove(...t.reduce(((e,t)=>e.concat(t.split(/\s+/))), [])),
            e) : (U.warn("removeClass was called with an element that doesn't exist"),
            null)
        }
        function Ee(e, t, i) {
            return "function" === typeof i && (i = i(e, t)),
            "boolean" !== typeof i && (i = void 0),
            t.split(/\s+/).forEach((t=>e.classList.toggle(t, i))),
            e
        }
        function we(e, t) {
            Object.getOwnPropertyNames(t).forEach((function(i) {
                const s = t[i];
                null === s || "undefined" === typeof s || !1 === s ? e.removeAttribute(i) : e.setAttribute(i, !0 === s ? "" : s)
            }
            ))
        }
        function xe(e) {
            const t = {}
              , i = ["autoplay", "controls", "playsinline", "loop", "muted", "default", "defaultMuted"];
            if (e && e.attributes && e.attributes.length > 0) {
                const s = e.attributes;
                for (let e = s.length - 1; e >= 0; e--) {
                    const n = s[e].name;
                    let r = s[e].value;
                    i.includes(n) && (r = null !== r),
                    t[n] = r
                }
            }
            return t
        }
        function Ie(e, t) {
            return e.getAttribute(t)
        }
        function Pe(e, t, i) {
            e.setAttribute(t, i)
        }
        function Ae(e, t) {
            e.removeAttribute(t)
        }
        function Le() {
            a().body.focus(),
            a().onselectstart = function() {
                return !1
            }
        }
        function De() {
            a().onselectstart = function() {
                return !0
            }
        }
        function Oe(e) {
            if (e && e.getBoundingClientRect && e.parentNode) {
                const t = e.getBoundingClientRect()
                  , i = {};
                return ["bottom", "height", "left", "right", "top", "width"].forEach((e=>{
                    void 0 !== t[e] && (i[e] = t[e])
                }
                )),
                i.height || (i.height = parseFloat(Ve(e, "height"))),
                i.width || (i.width = parseFloat(Ve(e, "width"))),
                i
            }
        }
        function Me(e) {
            if (!e || e && !e.offsetParent)
                return {
                    left: 0,
                    top: 0,
                    width: 0,
                    height: 0
                };
            const t = e.offsetWidth
              , i = e.offsetHeight;
            let s = 0
              , n = 0;
            for (; e.offsetParent && e !== a()[L.fullscreenElement]; )
                s += e.offsetLeft,
                n += e.offsetTop,
                e = e.offsetParent;
            return {
                left: s,
                top: n,
                width: t,
                height: i
            }
        }
        function Re(e, t) {
            const i = {
                x: 0,
                y: 0
            };
            if (ue) {
                let t = e;
                for (; t && "html" !== t.nodeName.toLowerCase(); ) {
                    const e = Ve(t, "transform");
                    if (/^matrix/.test(e)) {
                        const t = e.slice(7, -1).split(/,\s/).map(Number);
                        i.x += t[4],
                        i.y += t[5]
                    } else if (/^matrix3d/.test(e)) {
                        const t = e.slice(9, -1).split(/,\s/).map(Number);
                        i.x += t[12],
                        i.y += t[13]
                    }
                    t = t.parentNode
                }
            }
            const s = {}
              , n = Me(t.target)
              , r = Me(e)
              , a = r.width
              , o = r.height;
            let l = t.offsetY - (r.top - n.top)
              , h = t.offsetX - (r.left - n.left);
            return t.changedTouches && (h = t.changedTouches[0].pageX - r.left,
            l = t.changedTouches[0].pageY + r.top,
            ue && (h -= i.x,
            l -= i.y)),
            s.y = 1 - Math.max(0, Math.min(1, l / o)),
            s.x = Math.max(0, Math.min(1, h / a)),
            s
        }
        function Ue(e) {
            return q(e) && 3 === e.nodeType
        }
        function Be(e) {
            for (; e.firstChild; )
                e.removeChild(e.firstChild);
            return e
        }
        function Ne(e) {
            return "function" === typeof e && (e = e()),
            (Array.isArray(e) ? e : [e]).map((e=>("function" === typeof e && (e = e()),
            fe(e) || Ue(e) ? e : "string" === typeof e && /\S/.test(e) ? a().createTextNode(e) : void 0))).filter((e=>e))
        }
        function Fe(e, t) {
            return Ne(t).forEach((t=>e.appendChild(t))),
            e
        }
        function je(e, t) {
            return Fe(Be(e), t)
        }
        function $e(e) {
            return void 0 === e.button && void 0 === e.buttons || (0 === e.button && void 0 === e.buttons || ("mouseup" === e.type && 0 === e.button && 0 === e.buttons || 0 === e.button && 1 === e.buttons))
        }
        const qe = ye("querySelector")
          , He = ye("querySelectorAll");
        function Ve(e, t) {
            if (!e || !t)
                return "";
            if ("function" === typeof n().getComputedStyle) {
                let s;
                try {
                    s = n().getComputedStyle(e)
                } catch (i) {
                    return ""
                }
                return s ? s.getPropertyValue(t) || s[t] : ""
            }
            return ""
        }
        function ze(e) {
            [...a().styleSheets].forEach((t=>{
                try {
                    const i = [...t.cssRules].map((e=>e.cssText)).join("")
                      , s = a().createElement("style");
                    s.textContent = i,
                    e.document.head.appendChild(s)
                } catch (i) {
                    const s = a().createElement("link");
                    s.rel = "stylesheet",
                    s.type = t.type,
                    s.media = t.media.mediaText,
                    s.href = t.href,
                    e.document.head.appendChild(s)
                }
            }
            ))
        }
        var We = Object.freeze({
            __proto__: null,
            isReal: ge,
            isEl: fe,
            isInFrame: _e,
            createEl: ve,
            textContent: Te,
            prependTo: be,
            hasClass: Se,
            addClass: ke,
            removeClass: Ce,
            toggleClass: Ee,
            setAttributes: we,
            getAttributes: xe,
            getAttribute: Ie,
            setAttribute: Pe,
            removeAttribute: Ae,
            blockTextSelection: Le,
            unblockTextSelection: De,
            getBoundingClientRect: Oe,
            findPosition: Me,
            getPointerPosition: Re,
            isTextNode: Ue,
            emptyEl: Be,
            normalizeContent: Ne,
            appendContent: Fe,
            insertContent: je,
            isSingleLeftClick: $e,
            $: qe,
            $$: He,
            computedStyle: Ve,
            copyStyleSheetsToWindow: ze
        });
        let Ge, Ke = !1;
        const Qe = function() {
            if (!1 === Ge.options.autoSetup)
                return;
            const e = Array.prototype.slice.call(a().getElementsByTagName("video"))
              , t = Array.prototype.slice.call(a().getElementsByTagName("audio"))
              , i = Array.prototype.slice.call(a().getElementsByTagName("video-js"))
              , s = e.concat(t, i);
            if (s && s.length > 0)
                for (let n = 0, r = s.length; n < r; n++) {
                    const e = s[n];
                    if (!e || !e.getAttribute) {
                        Xe(1);
                        break
                    }
                    if (void 0 === e.player) {
                        null !== e.getAttribute("data-setup") && Ge(e)
                    }
                }
            else
                Ke || Xe(1)
        };
        function Xe(e, t) {
            ge() && (t && (Ge = t),
            n().setTimeout(Qe, e))
        }
        function Ye() {
            Ke = !0,
            n().removeEventListener("load", Ye)
        }
        ge() && ("complete" === a().readyState ? Ye() : n().addEventListener("load", Ye));
        const Je = function(e) {
            const t = a().createElement("style");
            return t.className = e,
            t
        }
          , Ze = function(e, t) {
            e.styleSheet ? e.styleSheet.cssText = t : e.textContent = t
        };
        var et = new WeakMap;
        let tt, it = 3;
        function st() {
            return it++
        }
        function nt(e, t) {
            if (!et.has(e))
                return;
            const i = et.get(e);
            0 === i.handlers[t].length && (delete i.handlers[t],
            e.removeEventListener ? e.removeEventListener(t, i.dispatcher, !1) : e.detachEvent && e.detachEvent("on" + t, i.dispatcher)),
            Object.getOwnPropertyNames(i.handlers).length <= 0 && (delete i.handlers,
            delete i.dispatcher,
            delete i.disabled),
            0 === Object.getOwnPropertyNames(i).length && et.delete(e)
        }
        function rt(e, t, i, s) {
            i.forEach((function(i) {
                e(t, i, s)
            }
            ))
        }
        function at(e) {
            if (e.fixed_)
                return e;
            function t() {
                return !0
            }
            function i() {
                return !1
            }
            if (!e || !e.isPropagationStopped || !e.isImmediatePropagationStopped) {
                const s = e || n().event;
                e = {};
                for (const t in s)
                    "layerX" !== t && "layerY" !== t && "keyLocation" !== t && "webkitMovementX" !== t && "webkitMovementY" !== t && "path" !== t && ("returnValue" === t && s.preventDefault || (e[t] = s[t]));
                if (e.target || (e.target = e.srcElement || a()),
                e.relatedTarget || (e.relatedTarget = e.fromElement === e.target ? e.toElement : e.fromElement),
                e.preventDefault = function() {
                    s.preventDefault && s.preventDefault(),
                    e.returnValue = !1,
                    s.returnValue = !1,
                    e.defaultPrevented = !0
                }
                ,
                e.defaultPrevented = !1,
                e.stopPropagation = function() {
                    s.stopPropagation && s.stopPropagation(),
                    e.cancelBubble = !0,
                    s.cancelBubble = !0,
                    e.isPropagationStopped = t
                }
                ,
                e.isPropagationStopped = i,
                e.stopImmediatePropagation = function() {
                    s.stopImmediatePropagation && s.stopImmediatePropagation(),
                    e.isImmediatePropagationStopped = t,
                    e.stopPropagation()
                }
                ,
                e.isImmediatePropagationStopped = i,
                null !== e.clientX && void 0 !== e.clientX) {
                    const t = a().documentElement
                      , i = a().body;
                    e.pageX = e.clientX + (t && t.scrollLeft || i && i.scrollLeft || 0) - (t && t.clientLeft || i && i.clientLeft || 0),
                    e.pageY = e.clientY + (t && t.scrollTop || i && i.scrollTop || 0) - (t && t.clientTop || i && i.clientTop || 0)
                }
                e.which = e.charCode || e.keyCode,
                null !== e.button && void 0 !== e.button && (e.button = 1 & e.button ? 0 : 4 & e.button ? 1 : 2 & e.button ? 2 : 0)
            }
            return e.fixed_ = !0,
            e
        }
        const ot = ["touchstart", "touchmove"];
        function lt(e, t, i) {
            if (Array.isArray(t))
                return rt(lt, e, t, i);
            et.has(e) || et.set(e, {});
            const s = et.get(e);
            if (s.handlers || (s.handlers = {}),
            s.handlers[t] || (s.handlers[t] = []),
            i.guid || (i.guid = st()),
            s.handlers[t].push(i),
            s.dispatcher || (s.disabled = !1,
            s.dispatcher = function(t, i) {
                if (s.disabled)
                    return;
                t = at(t);
                const n = s.handlers[t.type];
                if (n) {
                    const s = n.slice(0);
                    for (let n = 0, a = s.length; n < a && !t.isImmediatePropagationStopped(); n++)
                        try {
                            s[n].call(e, t, i)
                        } catch (r) {
                            U.error(r)
                        }
                }
            }
            ),
            1 === s.handlers[t].length)
                if (e.addEventListener) {
                    let i = !1;
                    (function() {
                        if ("boolean" !== typeof tt) {
                            tt = !1;
                            try {
                                const e = Object.defineProperty({}, "passive", {
                                    get() {
                                        tt = !0
                                    }
                                });
                                n().addEventListener("test", null, e),
                                n().removeEventListener("test", null, e)
                            } catch (e) {}
                        }
                        return tt
                    }
                    )() && ot.indexOf(t) > -1 && (i = {
                        passive: !0
                    }),
                    e.addEventListener(t, s.dispatcher, i)
                } else
                    e.attachEvent && e.attachEvent("on" + t, s.dispatcher)
        }
        function ht(e, t, i) {
            if (!et.has(e))
                return;
            const s = et.get(e);
            if (!s.handlers)
                return;
            if (Array.isArray(t))
                return rt(ht, e, t, i);
            const n = function(e, t) {
                s.handlers[t] = [],
                nt(e, t)
            };
            if (void 0 === t) {
                for (const t in s.handlers)
                    Object.prototype.hasOwnProperty.call(s.handlers || {}, t) && n(e, t);
                return
            }
            const r = s.handlers[t];
            if (r)
                if (i) {
                    if (i.guid)
                        for (let e = 0; e < r.length; e++)
                            r[e].guid === i.guid && r.splice(e--, 1);
                    nt(e, t)
                } else
                    n(e, t)
        }
        function dt(e, t, i) {
            const s = et.has(e) ? et.get(e) : {}
              , n = e.parentNode || e.ownerDocument;
            if ("string" === typeof t ? t = {
                type: t,
                target: e
            } : t.target || (t.target = e),
            t = at(t),
            s.dispatcher && s.dispatcher.call(e, t, i),
            n && !t.isPropagationStopped() && !0 === t.bubbles)
                dt.call(null, n, t, i);
            else if (!n && !t.defaultPrevented && t.target && t.target[t.type]) {
                et.has(t.target) || et.set(t.target, {});
                const e = et.get(t.target);
                t.target[t.type] && (e.disabled = !0,
                "function" === typeof t.target[t.type] && t.target[t.type](),
                e.disabled = !1)
            }
            return !t.defaultPrevented
        }
        function ut(e, t, i) {
            if (Array.isArray(t))
                return rt(ut, e, t, i);
            const s = function() {
                ht(e, t, s),
                i.apply(this, arguments)
            };
            s.guid = i.guid = i.guid || st(),
            lt(e, t, s)
        }
        function ct(e, t, i) {
            const s = function() {
                ht(e, t, s),
                i.apply(this, arguments)
            };
            s.guid = i.guid = i.guid || st(),
            lt(e, t, s)
        }
        var pt = Object.freeze({
            __proto__: null,
            fixEvent: at,
            on: lt,
            off: ht,
            trigger: dt,
            one: ut,
            any: ct
        });
        const mt = 30
          , gt = function(e, t, i) {
            t.guid || (t.guid = st());
            const s = t.bind(e);
            return s.guid = i ? i + "_" + t.guid : t.guid,
            s
        }
          , ft = function(e, t) {
            let i = n().performance.now();
            return function(...s) {
                const r = n().performance.now();
                r - i >= t && (e(...s),
                i = r)
            }
        }
          , _t = function(e, t, i, s=n()) {
            let r;
            const a = function() {
                const n = this
                  , a = arguments;
                let o = function() {
                    r = null,
                    o = null,
                    i || e.apply(n, a)
                };
                !r && i && e.apply(n, a),
                s.clearTimeout(r),
                r = s.setTimeout(o, t)
            };
            return a.cancel = ()=>{
                s.clearTimeout(r),
                r = null
            }
            ,
            a
        };
        var yt = Object.freeze({
            __proto__: null,
            UPDATE_REFRESH_INTERVAL: mt,
            bind_: gt,
            throttle: ft,
            debounce: _t
        });
        let vt;
        class Tt {
            on(e, t) {
                const i = this.addEventListener;
                this.addEventListener = ()=>{}
                ,
                lt(this, e, t),
                this.addEventListener = i
            }
            off(e, t) {
                ht(this, e, t)
            }
            one(e, t) {
                const i = this.addEventListener;
                this.addEventListener = ()=>{}
                ,
                ut(this, e, t),
                this.addEventListener = i
            }
            any(e, t) {
                const i = this.addEventListener;
                this.addEventListener = ()=>{}
                ,
                ct(this, e, t),
                this.addEventListener = i
            }
            trigger(e) {
                const t = e.type || e;
                "string" === typeof e && (e = {
                    type: t
                }),
                e = at(e),
                this.allowedEvents_[t] && this["on" + t] && this["on" + t](e),
                dt(this, e)
            }
            queueTrigger(e) {
                vt || (vt = new Map);
                const t = e.type || e;
                let i = vt.get(this);
                i || (i = new Map,
                vt.set(this, i));
                const s = i.get(t);
                i.delete(t),
                n().clearTimeout(s);
                const r = n().setTimeout((()=>{
                    i.delete(t),
                    0 === i.size && (i = null,
                    vt.delete(this)),
                    this.trigger(e)
                }
                ), 0);
                i.set(t, r)
            }
        }
        Tt.prototype.allowedEvents_ = {},
        Tt.prototype.addEventListener = Tt.prototype.on,
        Tt.prototype.removeEventListener = Tt.prototype.off,
        Tt.prototype.dispatchEvent = Tt.prototype.trigger;
        const bt = e=>"function" === typeof e.name ? e.name() : "string" === typeof e.name ? e.name : e.name_ ? e.name_ : e.constructor && e.constructor.name ? e.constructor.name : typeof e
          , St = e=>e instanceof Tt || !!e.eventBusEl_ && ["on", "one", "off", "trigger"].every((t=>"function" === typeof e[t]))
          , kt = e=>"string" === typeof e && /\S/.test(e) || Array.isArray(e) && !!e.length
          , Ct = (e,t,i)=>{
            if (!e || !e.nodeName && !St(e))
                throw new Error(`Invalid target for ${bt(t)}#${i}; must be a DOM node or evented object.`)
        }
          , Et = (e,t,i)=>{
            if (!kt(e))
                throw new Error(`Invalid event type for ${bt(t)}#${i}; must be a non-empty string or array.`)
        }
          , wt = (e,t,i)=>{
            if ("function" !== typeof e)
                throw new Error(`Invalid listener for ${bt(t)}#${i}; must be a function.`)
        }
          , xt = (e,t,i)=>{
            const s = t.length < 3 || t[0] === e || t[0] === e.eventBusEl_;
            let n, r, a;
            return s ? (n = e.eventBusEl_,
            t.length >= 3 && t.shift(),
            [r,a] = t) : [n,r,a] = t,
            Ct(n, e, i),
            Et(r, e, i),
            wt(a, e, i),
            a = gt(e, a),
            {
                isTargetingSelf: s,
                target: n,
                type: r,
                listener: a
            }
        }
          , It = (e,t,i,s)=>{
            Ct(e, e, t),
            e.nodeName ? pt[t](e, i, s) : e[t](i, s)
        }
          , Pt = {
            on(...e) {
                const {isTargetingSelf: t, target: i, type: s, listener: n} = xt(this, e, "on");
                if (It(i, "on", s, n),
                !t) {
                    const e = ()=>this.off(i, s, n);
                    e.guid = n.guid;
                    const t = ()=>this.off("dispose", e);
                    t.guid = n.guid,
                    It(this, "on", "dispose", e),
                    It(i, "on", "dispose", t)
                }
            },
            one(...e) {
                const {isTargetingSelf: t, target: i, type: s, listener: n} = xt(this, e, "one");
                if (t)
                    It(i, "one", s, n);
                else {
                    const e = (...t)=>{
                        this.off(i, s, e),
                        n.apply(null, t)
                    }
                    ;
                    e.guid = n.guid,
                    It(i, "one", s, e)
                }
            },
            any(...e) {
                const {isTargetingSelf: t, target: i, type: s, listener: n} = xt(this, e, "any");
                if (t)
                    It(i, "any", s, n);
                else {
                    const e = (...t)=>{
                        this.off(i, s, e),
                        n.apply(null, t)
                    }
                    ;
                    e.guid = n.guid,
                    It(i, "any", s, e)
                }
            },
            off(e, t, i) {
                if (!e || kt(e))
                    ht(this.eventBusEl_, e, t);
                else {
                    const s = e
                      , n = t;
                    Ct(s, this, "off"),
                    Et(n, this, "off"),
                    wt(i, this, "off"),
                    i = gt(this, i),
                    this.off("dispose", i),
                    s.nodeName ? (ht(s, n, i),
                    ht(s, "dispose", i)) : St(s) && (s.off(n, i),
                    s.off("dispose", i))
                }
            },
            trigger(e, t) {
                Ct(this.eventBusEl_, this, "trigger");
                const i = e && "string" !== typeof e ? e.type : e;
                if (!kt(i))
                    throw new Error(`Invalid event type for ${bt(this)}#trigger; must be a non-empty string or object with a type key that has a non-empty value.`);
                return dt(this.eventBusEl_, e, t)
            }
        };
        function At(e, t={}) {
            const {eventBusKey: i} = t;
            if (i) {
                if (!e[i].nodeName)
                    throw new Error(`The eventBusKey "${i}" does not refer to an element.`);
                e.eventBusEl_ = e[i]
            } else
                e.eventBusEl_ = ve("span", {
                    className: "vjs-event-bus"
                });
            return Object.assign(e, Pt),
            e.eventedCallbacks && e.eventedCallbacks.forEach((e=>{
                e()
            }
            )),
            e.on("dispose", (()=>{
                e.off(),
                [e, e.el_, e.eventBusEl_].forEach((function(e) {
                    e && et.has(e) && et.delete(e)
                }
                )),
                n().setTimeout((()=>{
                    e.eventBusEl_ = null
                }
                ), 0)
            }
            )),
            e
        }
        const Lt = {
            state: {},
            setState(e) {
                let t;
                return "function" === typeof e && (e = e()),
                j(e, ((e,i)=>{
                    this.state[i] !== e && (t = t || {},
                    t[i] = {
                        from: this.state[i],
                        to: e
                    }),
                    this.state[i] = e
                }
                )),
                t && St(this) && this.trigger({
                    changes: t,
                    type: "statechanged"
                }),
                t
            }
        };
        function Dt(e, t) {
            return Object.assign(e, Lt),
            e.state = Object.assign({}, e.state, t),
            "function" === typeof e.handleStateChanged && St(e) && e.on("statechanged", e.handleStateChanged),
            e
        }
        const Ot = function(e) {
            return "string" !== typeof e ? e : e.replace(/./, (e=>e.toLowerCase()))
        }
          , Mt = function(e) {
            return "string" !== typeof e ? e : e.replace(/./, (e=>e.toUpperCase()))
        }
          , Rt = function(e, t) {
            return Mt(e) === Mt(t)
        };
        var Ut = Object.freeze({
            __proto__: null,
            toLowerCase: Ot,
            toTitleCase: Mt,
            titleCaseEquals: Rt
        });
        class Bt {
            constructor(e, t, i) {
                if (!e && this.play ? this.player_ = e = this : this.player_ = e,
                this.isDisposed_ = !1,
                this.parentComponent_ = null,
                this.options_ = V({}, this.options_),
                t = this.options_ = V(this.options_, t),
                this.id_ = t.id || t.el && t.el.id,
                !this.id_) {
                    const t = e && e.id && e.id() || "no_player";
                    this.id_ = `${t}_component_${st()}`
                }
                this.name_ = t.name || null,
                t.el ? this.el_ = t.el : !1 !== t.createEl && (this.el_ = this.createEl()),
                t.className && this.el_ && t.className.split(" ").forEach((e=>this.addClass(e))),
                ["on", "off", "one", "any", "trigger"].forEach((e=>{
                    this[e] = void 0
                }
                )),
                !1 !== t.evented && (At(this, {
                    eventBusKey: this.el_ ? "el_" : null
                }),
                this.handleLanguagechange = this.handleLanguagechange.bind(this),
                this.on(this.player_, "languagechange", this.handleLanguagechange)),
                Dt(this, this.constructor.defaultState),
                this.children_ = [],
                this.childIndex_ = {},
                this.childNameIndex_ = {},
                this.setTimeoutIds_ = new Set,
                this.setIntervalIds_ = new Set,
                this.rafIds_ = new Set,
                this.namedRafs_ = new Map,
                this.clearingTimersOnDispose_ = !1,
                !1 !== t.initChildren && this.initChildren(),
                this.ready(i),
                !1 !== t.reportTouchActivity && this.enableTouchActivity()
            }
            on(e, t) {}
            off(e, t) {}
            one(e, t) {}
            any(e, t) {}
            trigger(e, t) {}
            dispose(e={}) {
                if (!this.isDisposed_) {
                    if (this.readyQueue_ && (this.readyQueue_.length = 0),
                    this.trigger({
                        type: "dispose",
                        bubbles: !1
                    }),
                    this.isDisposed_ = !0,
                    this.children_)
                        for (let e = this.children_.length - 1; e >= 0; e--)
                            this.children_[e].dispose && this.children_[e].dispose();
                    this.children_ = null,
                    this.childIndex_ = null,
                    this.childNameIndex_ = null,
                    this.parentComponent_ = null,
                    this.el_ && (this.el_.parentNode && (e.restoreEl ? this.el_.parentNode.replaceChild(e.restoreEl, this.el_) : this.el_.parentNode.removeChild(this.el_)),
                    this.el_ = null),
                    this.player_ = null
                }
            }
            isDisposed() {
                return Boolean(this.isDisposed_)
            }
            player() {
                return this.player_
            }
            options(e) {
                return e ? (this.options_ = V(this.options_, e),
                this.options_) : this.options_
            }
            el() {
                return this.el_
            }
            createEl(e, t, i) {
                return ve(e, t, i)
            }
            localize(e, t, i=e) {
                const s = this.player_.language && this.player_.language()
                  , n = this.player_.languages && this.player_.languages()
                  , r = n && n[s]
                  , a = s && s.split("-")[0]
                  , o = n && n[a];
                let l = i;
                return r && r[e] ? l = r[e] : o && o[e] && (l = o[e]),
                t && (l = l.replace(/\{(\d+)\}/g, (function(e, i) {
                    const s = t[i - 1];
                    let n = s;
                    return "undefined" === typeof s && (n = e),
                    n
                }
                ))),
                l
            }
            handleLanguagechange() {}
            contentEl() {
                return this.contentEl_ || this.el_
            }
            id() {
                return this.id_
            }
            name() {
                return this.name_
            }
            children() {
                return this.children_
            }
            getChildById(e) {
                return this.childIndex_[e]
            }
            getChild(e) {
                if (e)
                    return this.childNameIndex_[e]
            }
            getDescendant(...e) {
                e = e.reduce(((e,t)=>e.concat(t)), []);
                let t = this;
                for (let i = 0; i < e.length; i++)
                    if (t = t.getChild(e[i]),
                    !t || !t.getChild)
                        return;
                return t
            }
            setIcon(e, t=this.el()) {
                if (!this.player_.options_.experimentalSvgIcons)
                    return;
                const i = "http://www.w3.org/2000/svg"
                  , s = ve("span", {
                    className: "vjs-icon-placeholder vjs-svg-icon"
                }, {
                    "aria-hidden": "true"
                })
                  , n = a().createElementNS(i, "svg");
                n.setAttributeNS(null, "viewBox", "0 0 512 512");
                const r = a().createElementNS(i, "use");
                return n.appendChild(r),
                r.setAttributeNS(null, "href", `#vjs-icon-${e}`),
                s.appendChild(n),
                this.iconIsSet_ ? t.replaceChild(s, t.querySelector(".vjs-icon-placeholder")) : t.appendChild(s),
                this.iconIsSet_ = !0,
                s
            }
            addChild(e, t={}, i=this.children_.length) {
                let s, n;
                if ("string" === typeof e) {
                    n = Mt(e);
                    const i = t.componentClass || n;
                    t.name = n;
                    const r = Bt.getComponent(i);
                    if (!r)
                        throw new Error(`Component ${i} does not exist`);
                    if ("function" !== typeof r)
                        return null;
                    s = new r(this.player_ || this,t)
                } else
                    s = e;
                if (s.parentComponent_ && s.parentComponent_.removeChild(s),
                this.children_.splice(i, 0, s),
                s.parentComponent_ = this,
                "function" === typeof s.id && (this.childIndex_[s.id()] = s),
                n = n || s.name && Mt(s.name()),
                n && (this.childNameIndex_[n] = s,
                this.childNameIndex_[Ot(n)] = s),
                "function" === typeof s.el && s.el()) {
                    let e = null;
                    this.children_[i + 1] && (this.children_[i + 1].el_ ? e = this.children_[i + 1].el_ : fe(this.children_[i + 1]) && (e = this.children_[i + 1])),
                    this.contentEl().insertBefore(s.el(), e)
                }
                return s
            }
            removeChild(e) {
                if ("string" === typeof e && (e = this.getChild(e)),
                !e || !this.children_)
                    return;
                let t = !1;
                for (let s = this.children_.length - 1; s >= 0; s--)
                    if (this.children_[s] === e) {
                        t = !0,
                        this.children_.splice(s, 1);
                        break
                    }
                if (!t)
                    return;
                e.parentComponent_ = null,
                this.childIndex_[e.id()] = null,
                this.childNameIndex_[Mt(e.name())] = null,
                this.childNameIndex_[Ot(e.name())] = null;
                const i = e.el();
                i && i.parentNode === this.contentEl() && this.contentEl().removeChild(e.el())
            }
            initChildren() {
                const e = this.options_.children;
                if (e) {
                    const t = this.options_
                      , i = e=>{
                        const i = e.name;
                        let s = e.opts;
                        if (void 0 !== t[i] && (s = t[i]),
                        !1 === s)
                            return;
                        !0 === s && (s = {}),
                        s.playerOptions = this.options_.playerOptions;
                        const n = this.addChild(i, s);
                        n && (this[i] = n)
                    }
                    ;
                    let s;
                    const n = Bt.getComponent("Tech");
                    s = Array.isArray(e) ? e : Object.keys(e),
                    s.concat(Object.keys(this.options_).filter((function(e) {
                        return !s.some((function(t) {
                            return "string" === typeof t ? e === t : e === t.name
                        }
                        ))
                    }
                    ))).map((t=>{
                        let i, s;
                        return "string" === typeof t ? (i = t,
                        s = e[i] || this.options_[i] || {}) : (i = t.name,
                        s = t),
                        {
                            name: i,
                            opts: s
                        }
                    }
                    )).filter((e=>{
                        const t = Bt.getComponent(e.opts.componentClass || Mt(e.name));
                        return t && !n.isTech(t)
                    }
                    )).forEach(i)
                }
            }
            buildCSSClass() {
                return ""
            }
            ready(e, t=!1) {
                if (e)
                    return this.isReady_ ? void (t ? e.call(this) : this.setTimeout(e, 1)) : (this.readyQueue_ = this.readyQueue_ || [],
                    void this.readyQueue_.push(e))
            }
            triggerReady() {
                this.isReady_ = !0,
                this.setTimeout((function() {
                    const e = this.readyQueue_;
                    this.readyQueue_ = [],
                    e && e.length > 0 && e.forEach((function(e) {
                        e.call(this)
                    }
                    ), this),
                    this.trigger("ready")
                }
                ), 1)
            }
            $(e, t) {
                return qe(e, t || this.contentEl())
            }
            $$(e, t) {
                return He(e, t || this.contentEl())
            }
            hasClass(e) {
                return Se(this.el_, e)
            }
            addClass(...e) {
                ke(this.el_, ...e)
            }
            removeClass(...e) {
                Ce(this.el_, ...e)
            }
            toggleClass(e, t) {
                Ee(this.el_, e, t)
            }
            show() {
                this.removeClass("vjs-hidden")
            }
            hide() {
                this.addClass("vjs-hidden")
            }
            lockShowing() {
                this.addClass("vjs-lock-showing")
            }
            unlockShowing() {
                this.removeClass("vjs-lock-showing")
            }
            getAttribute(e) {
                return Ie(this.el_, e)
            }
            setAttribute(e, t) {
                Pe(this.el_, e, t)
            }
            removeAttribute(e) {
                Ae(this.el_, e)
            }
            width(e, t) {
                return this.dimension("width", e, t)
            }
            height(e, t) {
                return this.dimension("height", e, t)
            }
            dimensions(e, t) {
                this.width(e, !0),
                this.height(t)
            }
            dimension(e, t, i) {
                if (void 0 !== t)
                    return null !== t && t === t || (t = 0),
                    -1 !== ("" + t).indexOf("%") || -1 !== ("" + t).indexOf("px") ? this.el_.style[e] = t : this.el_.style[e] = "auto" === t ? "" : t + "px",
                    void (i || this.trigger("componentresize"));
                if (!this.el_)
                    return 0;
                const s = this.el_.style[e]
                  , n = s.indexOf("px");
                return -1 !== n ? parseInt(s.slice(0, n), 10) : parseInt(this.el_["offset" + Mt(e)], 10)
            }
            currentDimension(e) {
                let t = 0;
                if ("width" !== e && "height" !== e)
                    throw new Error("currentDimension only accepts width or height value");
                if (t = Ve(this.el_, e),
                t = parseFloat(t),
                0 === t || isNaN(t)) {
                    const i = `offset${Mt(e)}`;
                    t = this.el_[i]
                }
                return t
            }
            currentDimensions() {
                return {
                    width: this.currentDimension("width"),
                    height: this.currentDimension("height")
                }
            }
            currentWidth() {
                return this.currentDimension("width")
            }
            currentHeight() {
                return this.currentDimension("height")
            }
            focus() {
                this.el_.focus()
            }
            blur() {
                this.el_.blur()
            }
            handleKeyDown(e) {
                this.player_ && (l().isEventKey(e, "Tab") || e.stopPropagation(),
                this.player_.handleKeyDown(e))
            }
            handleKeyPress(e) {
                this.handleKeyDown(e)
            }
            emitTapEvents() {
                let e = 0
                  , t = null;
                let i;
                this.on("touchstart", (function(s) {
                    1 === s.touches.length && (t = {
                        pageX: s.touches[0].pageX,
                        pageY: s.touches[0].pageY
                    },
                    e = n().performance.now(),
                    i = !0)
                }
                )),
                this.on("touchmove", (function(e) {
                    if (e.touches.length > 1)
                        i = !1;
                    else if (t) {
                        const s = e.touches[0].pageX - t.pageX
                          , n = e.touches[0].pageY - t.pageY;
                        Math.sqrt(s * s + n * n) > 10 && (i = !1)
                    }
                }
                ));
                const s = function() {
                    i = !1
                };
                this.on("touchleave", s),
                this.on("touchcancel", s),
                this.on("touchend", (function(s) {
                    if (t = null,
                    !0 === i) {
                        n().performance.now() - e < 200 && (s.preventDefault(),
                        this.trigger("tap"))
                    }
                }
                ))
            }
            enableTouchActivity() {
                if (!this.player() || !this.player().reportUserActivity)
                    return;
                const e = gt(this.player(), this.player().reportUserActivity);
                let t;
                this.on("touchstart", (function() {
                    e(),
                    this.clearInterval(t),
                    t = this.setInterval(e, 250)
                }
                ));
                const i = function(i) {
                    e(),
                    this.clearInterval(t)
                };
                this.on("touchmove", e),
                this.on("touchend", i),
                this.on("touchcancel", i)
            }
            setTimeout(e, t) {
                var i;
                return e = gt(this, e),
                this.clearTimersOnDispose_(),
                i = n().setTimeout((()=>{
                    this.setTimeoutIds_.has(i) && this.setTimeoutIds_.delete(i),
                    e()
                }
                ), t),
                this.setTimeoutIds_.add(i),
                i
            }
            clearTimeout(e) {
                return this.setTimeoutIds_.has(e) && (this.setTimeoutIds_.delete(e),
                n().clearTimeout(e)),
                e
            }
            setInterval(e, t) {
                e = gt(this, e),
                this.clearTimersOnDispose_();
                const i = n().setInterval(e, t);
                return this.setIntervalIds_.add(i),
                i
            }
            clearInterval(e) {
                return this.setIntervalIds_.has(e) && (this.setIntervalIds_.delete(e),
                n().clearInterval(e)),
                e
            }
            requestAnimationFrame(e) {
                var t;
                return this.clearTimersOnDispose_(),
                e = gt(this, e),
                t = n().requestAnimationFrame((()=>{
                    this.rafIds_.has(t) && this.rafIds_.delete(t),
                    e()
                }
                )),
                this.rafIds_.add(t),
                t
            }
            requestNamedAnimationFrame(e, t) {
                if (this.namedRafs_.has(e))
                    return;
                this.clearTimersOnDispose_(),
                t = gt(this, t);
                const i = this.requestAnimationFrame((()=>{
                    t(),
                    this.namedRafs_.has(e) && this.namedRafs_.delete(e)
                }
                ));
                return this.namedRafs_.set(e, i),
                e
            }
            cancelNamedAnimationFrame(e) {
                this.namedRafs_.has(e) && (this.cancelAnimationFrame(this.namedRafs_.get(e)),
                this.namedRafs_.delete(e))
            }
            cancelAnimationFrame(e) {
                return this.rafIds_.has(e) && (this.rafIds_.delete(e),
                n().cancelAnimationFrame(e)),
                e
            }
            clearTimersOnDispose_() {
                this.clearingTimersOnDispose_ || (this.clearingTimersOnDispose_ = !0,
                this.one("dispose", (()=>{
                    [["namedRafs_", "cancelNamedAnimationFrame"], ["rafIds_", "cancelAnimationFrame"], ["setTimeoutIds_", "clearTimeout"], ["setIntervalIds_", "clearInterval"]].forEach((([e,t])=>{
                        this[e].forEach(((e,i)=>this[t](i)))
                    }
                    )),
                    this.clearingTimersOnDispose_ = !1
                }
                )))
            }
            static registerComponent(e, t) {
                if ("string" !== typeof e || !e)
                    throw new Error(`Illegal component name, "${e}"; must be a non-empty string.`);
                const i = Bt.getComponent("Tech")
                  , s = i && i.isTech(t)
                  , n = Bt === t || Bt.prototype.isPrototypeOf(t.prototype);
                if (s || !n) {
                    let t;
                    throw t = s ? "techs must be registered using Tech.registerTech()" : "must be a Component subclass",
                    new Error(`Illegal component, "${e}"; ${t}.`)
                }
                e = Mt(e),
                Bt.components_ || (Bt.components_ = {});
                const r = Bt.getComponent("Player");
                if ("Player" === e && r && r.players) {
                    const e = r.players
                      , t = Object.keys(e);
                    if (e && t.length > 0 && t.map((t=>e[t])).every(Boolean))
                        throw new Error("Can not register Player component after player has been created.")
                }
                return Bt.components_[e] = t,
                Bt.components_[Ot(e)] = t,
                t
            }
            static getComponent(e) {
                if (e && Bt.components_)
                    return Bt.components_[e]
            }
        }
        function Nt(e, t, i, s) {
            return function(e, t, i) {
                if ("number" !== typeof t || t < 0 || t > i)
                    throw new Error(`Failed to execute '${e}' on 'TimeRanges': The index provided (${t}) is non-numeric or out of bounds (0-${i}).`)
            }(e, s, i.length - 1),
            i[s][t]
        }
        function Ft(e) {
            let t;
            return t = void 0 === e || 0 === e.length ? {
                length: 0,
                start() {
                    throw new Error("This TimeRanges object is empty")
                },
                end() {
                    throw new Error("This TimeRanges object is empty")
                }
            } : {
                length: e.length,
                start: Nt.bind(null, "start", 0, e),
                end: Nt.bind(null, "end", 1, e)
            },
            n().Symbol && n().Symbol.iterator && (t[n().Symbol.iterator] = ()=>(e || []).values()),
            t
        }
        function jt(e, t) {
            return Array.isArray(e) ? Ft(e) : void 0 === e || void 0 === t ? Ft() : Ft([[e, t]])
        }
        Bt.registerComponent("Component", Bt);
        const $t = function(e, t) {
            e = e < 0 ? 0 : e;
            let i = Math.floor(e % 60)
              , s = Math.floor(e / 60 % 60)
              , n = Math.floor(e / 3600);
            const r = Math.floor(t / 60 % 60)
              , a = Math.floor(t / 3600);
            return (isNaN(e) || e === 1 / 0) && (n = s = i = "-"),
            n = n > 0 || a > 0 ? n + ":" : "",
            s = ((n || r >= 10) && s < 10 ? "0" + s : s) + ":",
            i = i < 10 ? "0" + i : i,
            n + s + i
        };
        let qt = $t;
        function Ht(e) {
            qt = e
        }
        function Vt() {
            qt = $t
        }
        function zt(e, t=e) {
            return qt(e, t)
        }
        var Wt = Object.freeze({
            __proto__: null,
            createTimeRanges: jt,
            createTimeRange: jt,
            setFormatTime: Ht,
            resetFormatTime: Vt,
            formatTime: zt
        });
        function Gt(e, t) {
            let i, s, n = 0;
            if (!t)
                return 0;
            e && e.length || (e = jt(0, 0));
            for (let r = 0; r < e.length; r++)
                i = e.start(r),
                s = e.end(r),
                s > t && (s = t),
                n += s - i;
            return n / t
        }
        function Kt(e) {
            if (e instanceof Kt)
                return e;
            "number" === typeof e ? this.code = e : "string" === typeof e ? this.message = e : q(e) && ("number" === typeof e.code && (this.code = e.code),
            Object.assign(this, e)),
            this.message || (this.message = Kt.defaultMessages[this.code] || "")
        }
        Kt.prototype.code = 0,
        Kt.prototype.message = "",
        Kt.prototype.status = null,
        Kt.errorTypes = ["MEDIA_ERR_CUSTOM", "MEDIA_ERR_ABORTED", "MEDIA_ERR_NETWORK", "MEDIA_ERR_DECODE", "MEDIA_ERR_SRC_NOT_SUPPORTED", "MEDIA_ERR_ENCRYPTED"],
        Kt.defaultMessages = {
            1: "You aborted the media playback",
            2: "A network error caused the media download to fail part-way.",
            3: "The media playback was aborted due to a corruption problem or because the media used features your browser did not support.",
            4: "The media could not be loaded, either because the server or network failed or because the format is not supported.",
            5: "The media is encrypted and we do not have the keys to decrypt it."
        };
        for (let tl = 0; tl < Kt.errorTypes.length; tl++)
            Kt[Kt.errorTypes[tl]] = tl,
            Kt.prototype[Kt.errorTypes[tl]] = tl;
        function Qt(e) {
            return void 0 !== e && null !== e && "function" === typeof e.then
        }
        function Xt(e) {
            Qt(e) && e.then(null, (e=>{}
            ))
        }
        const Yt = function(e) {
            return ["kind", "label", "language", "id", "inBandMetadataTrackDispatchType", "mode", "src"].reduce(((t,i,s)=>(e[i] && (t[i] = e[i]),
            t)), {
                cues: e.cues && Array.prototype.map.call(e.cues, (function(e) {
                    return {
                        startTime: e.startTime,
                        endTime: e.endTime,
                        text: e.text,
                        id: e.id
                    }
                }
                ))
            })
        };
        var Jt = function(e) {
            const t = e.$$("track")
              , i = Array.prototype.map.call(t, (e=>e.track));
            return Array.prototype.map.call(t, (function(e) {
                const t = Yt(e.track);
                return e.src && (t.src = e.src),
                t
            }
            )).concat(Array.prototype.filter.call(e.textTracks(), (function(e) {
                return -1 === i.indexOf(e)
            }
            )).map(Yt))
        }
          , Zt = function(e, t) {
            return e.forEach((function(e) {
                const i = t.addRemoteTextTrack(e).track;
                !e.src && e.cues && e.cues.forEach((e=>i.addCue(e)))
            }
            )),
            t.textTracks()
        };
        class ei extends Bt {
            constructor(e, t) {
                super(e, t),
                this.handleKeyDown_ = e=>this.handleKeyDown(e),
                this.close_ = e=>this.close(e),
                this.opened_ = this.hasBeenOpened_ = this.hasBeenFilled_ = !1,
                this.closeable(!this.options_.uncloseable),
                this.content(this.options_.content),
                this.contentEl_ = ve("div", {
                    className: "vjs-modal-dialog-content"
                }, {
                    role: "document"
                }),
                this.descEl_ = ve("p", {
                    className: "vjs-modal-dialog-description vjs-control-text",
                    id: this.el().getAttribute("aria-describedby")
                }),
                Te(this.descEl_, this.description()),
                this.el_.appendChild(this.descEl_),
                this.el_.appendChild(this.contentEl_)
            }
            createEl() {
                return super.createEl("div", {
                    className: this.buildCSSClass(),
                    tabIndex: -1
                }, {
                    "aria-describedby": `${this.id()}_description`,
                    "aria-hidden": "true",
                    "aria-label": this.label(),
                    role: "dialog"
                })
            }
            dispose() {
                this.contentEl_ = null,
                this.descEl_ = null,
                this.previouslyActiveEl_ = null,
                super.dispose()
            }
            buildCSSClass() {
                return `vjs-modal-dialog vjs-hidden ${super.buildCSSClass()}`
            }
            label() {
                return this.localize(this.options_.label || "Modal Window")
            }
            description() {
                let e = this.options_.description || this.localize("This is a modal window.");
                return this.closeable() && (e += " " + this.localize("This modal can be closed by pressing the Escape key or activating the close button.")),
                e
            }
            open() {
                if (!this.opened_) {
                    const e = this.player();
                    this.trigger("beforemodalopen"),
                    this.opened_ = !0,
                    (this.options_.fillAlways || !this.hasBeenOpened_ && !this.hasBeenFilled_) && this.fill(),
                    this.wasPlaying_ = !e.paused(),
                    this.options_.pauseOnOpen && this.wasPlaying_ && e.pause(),
                    this.on("keydown", this.handleKeyDown_),
                    this.hadControls_ = e.controls(),
                    e.controls(!1),
                    this.show(),
                    this.conditionalFocus_(),
                    this.el().setAttribute("aria-hidden", "false"),
                    this.trigger("modalopen"),
                    this.hasBeenOpened_ = !0
                }
            }
            opened(e) {
                return "boolean" === typeof e && this[e ? "open" : "close"](),
                this.opened_
            }
            close() {
                if (!this.opened_)
                    return;
                const e = this.player();
                this.trigger("beforemodalclose"),
                this.opened_ = !1,
                this.wasPlaying_ && this.options_.pauseOnOpen && e.play(),
                this.off("keydown", this.handleKeyDown_),
                this.hadControls_ && e.controls(!0),
                this.hide(),
                this.el().setAttribute("aria-hidden", "true"),
                this.trigger("modalclose"),
                this.conditionalBlur_(),
                this.options_.temporary && this.dispose()
            }
            closeable(e) {
                if ("boolean" === typeof e) {
                    const t = this.closeable_ = !!e;
                    let i = this.getChild("closeButton");
                    if (t && !i) {
                        const e = this.contentEl_;
                        this.contentEl_ = this.el_,
                        i = this.addChild("closeButton", {
                            controlText: "Close Modal Dialog"
                        }),
                        this.contentEl_ = e,
                        this.on(i, "close", this.close_)
                    }
                    !t && i && (this.off(i, "close", this.close_),
                    this.removeChild(i),
                    i.dispose())
                }
                return this.closeable_
            }
            fill() {
                this.fillWith(this.content())
            }
            fillWith(e) {
                const t = this.contentEl()
                  , i = t.parentNode
                  , s = t.nextSibling;
                this.trigger("beforemodalfill"),
                this.hasBeenFilled_ = !0,
                i.removeChild(t),
                this.empty(),
                je(t, e),
                this.trigger("modalfill"),
                s ? i.insertBefore(t, s) : i.appendChild(t);
                const n = this.getChild("closeButton");
                n && i.appendChild(n.el_)
            }
            empty() {
                this.trigger("beforemodalempty"),
                Be(this.contentEl()),
                this.trigger("modalempty")
            }
            content(e) {
                return "undefined" !== typeof e && (this.content_ = e),
                this.content_
            }
            conditionalFocus_() {
                const e = a().activeElement
                  , t = this.player_.el_;
                this.previouslyActiveEl_ = null,
                (t.contains(e) || t === e) && (this.previouslyActiveEl_ = e,
                this.focus())
            }
            conditionalBlur_() {
                this.previouslyActiveEl_ && (this.previouslyActiveEl_.focus(),
                this.previouslyActiveEl_ = null)
            }
            handleKeyDown(e) {
                if (e.stopPropagation(),
                l().isEventKey(e, "Escape") && this.closeable())
                    return e.preventDefault(),
                    void this.close();
                if (!l().isEventKey(e, "Tab"))
                    return;
                const t = this.focusableEls_()
                  , i = this.el_.querySelector(":focus");
                let s;
                for (let n = 0; n < t.length; n++)
                    if (i === t[n]) {
                        s = n;
                        break
                    }
                a().activeElement === this.el_ && (s = 0),
                e.shiftKey && 0 === s ? (t[t.length - 1].focus(),
                e.preventDefault()) : e.shiftKey || s !== t.length - 1 || (t[0].focus(),
                e.preventDefault())
            }
            focusableEls_() {
                const e = this.el_.querySelectorAll("*");
                return Array.prototype.filter.call(e, (e=>(e instanceof n().HTMLAnchorElement || e instanceof n().HTMLAreaElement) && e.hasAttribute("href") || (e instanceof n().HTMLInputElement || e instanceof n().HTMLSelectElement || e instanceof n().HTMLTextAreaElement || e instanceof n().HTMLButtonElement) && !e.hasAttribute("disabled") || e instanceof n().HTMLIFrameElement || e instanceof n().HTMLObjectElement || e instanceof n().HTMLEmbedElement || e.hasAttribute("tabindex") && -1 !== e.getAttribute("tabindex") || e.hasAttribute("contenteditable")))
            }
        }
        ei.prototype.options_ = {
            pauseOnOpen: !0,
            temporary: !0
        },
        Bt.registerComponent("ModalDialog", ei);
        class ti extends Tt {
            constructor(e=[]) {
                super(),
                this.tracks_ = [],
                Object.defineProperty(this, "length", {
                    get() {
                        return this.tracks_.length
                    }
                });
                for (let t = 0; t < e.length; t++)
                    this.addTrack(e[t])
            }
            addTrack(e) {
                const t = this.tracks_.length;
                "" + t in this || Object.defineProperty(this, t, {
                    get() {
                        return this.tracks_[t]
                    }
                }),
                -1 === this.tracks_.indexOf(e) && (this.tracks_.push(e),
                this.trigger({
                    track: e,
                    type: "addtrack",
                    target: this
                })),
                e.labelchange_ = ()=>{
                    this.trigger({
                        track: e,
                        type: "labelchange",
                        target: this
                    })
                }
                ,
                St(e) && e.addEventListener("labelchange", e.labelchange_)
            }
            removeTrack(e) {
                let t;
                for (let i = 0, s = this.length; i < s; i++)
                    if (this[i] === e) {
                        t = this[i],
                        t.off && t.off(),
                        this.tracks_.splice(i, 1);
                        break
                    }
                t && this.trigger({
                    track: t,
                    type: "removetrack",
                    target: this
                })
            }
            getTrackById(e) {
                let t = null;
                for (let i = 0, s = this.length; i < s; i++) {
                    const s = this[i];
                    if (s.id === e) {
                        t = s;
                        break
                    }
                }
                return t
            }
        }
        ti.prototype.allowedEvents_ = {
            change: "change",
            addtrack: "addtrack",
            removetrack: "removetrack",
            labelchange: "labelchange"
        };
        for (const tl in ti.prototype.allowedEvents_)
            ti.prototype["on" + tl] = null;
        const ii = function(e, t) {
            for (let i = 0; i < e.length; i++)
                Object.keys(e[i]).length && t.id !== e[i].id && (e[i].enabled = !1)
        };
        const si = function(e, t) {
            for (let i = 0; i < e.length; i++)
                Object.keys(e[i]).length && t.id !== e[i].id && (e[i].selected = !1)
        };
        class ni extends ti {
            addTrack(e) {
                super.addTrack(e),
                this.queueChange_ || (this.queueChange_ = ()=>this.queueTrigger("change")),
                this.triggerSelectedlanguagechange || (this.triggerSelectedlanguagechange_ = ()=>this.trigger("selectedlanguagechange")),
                e.addEventListener("modechange", this.queueChange_);
                -1 === ["metadata", "chapters"].indexOf(e.kind) && e.addEventListener("modechange", this.triggerSelectedlanguagechange_)
            }
            removeTrack(e) {
                super.removeTrack(e),
                e.removeEventListener && (this.queueChange_ && e.removeEventListener("modechange", this.queueChange_),
                this.selectedlanguagechange_ && e.removeEventListener("modechange", this.triggerSelectedlanguagechange_))
            }
        }
        class ri {
            constructor(e) {
                ri.prototype.setCues_.call(this, e),
                Object.defineProperty(this, "length", {
                    get() {
                        return this.length_
                    }
                })
            }
            setCues_(e) {
                const t = this.length || 0;
                let i = 0;
                const s = e.length;
                this.cues_ = e,
                this.length_ = e.length;
                const n = function(e) {
                    "" + e in this || Object.defineProperty(this, "" + e, {
                        get() {
                            return this.cues_[e]
                        }
                    })
                };
                if (t < s)
                    for (i = t; i < s; i++)
                        n.call(this, i)
            }
            getCueById(e) {
                let t = null;
                for (let i = 0, s = this.length; i < s; i++) {
                    const s = this[i];
                    if (s.id === e) {
                        t = s;
                        break
                    }
                }
                return t
            }
        }
        const ai = {
            alternative: "alternative",
            captions: "captions",
            main: "main",
            sign: "sign",
            subtitles: "subtitles",
            commentary: "commentary"
        }
          , oi = {
            alternative: "alternative",
            descriptions: "descriptions",
            main: "main",
            "main-desc": "main-desc",
            translation: "translation",
            commentary: "commentary"
        }
          , li = {
            subtitles: "subtitles",
            captions: "captions",
            descriptions: "descriptions",
            chapters: "chapters",
            metadata: "metadata"
        }
          , hi = {
            disabled: "disabled",
            hidden: "hidden",
            showing: "showing"
        };
        class di extends Tt {
            constructor(e={}) {
                super();
                const t = {
                    id: e.id || "vjs_track_" + st(),
                    kind: e.kind || "",
                    language: e.language || ""
                };
                let i = e.label || "";
                for (const s in t)
                    Object.defineProperty(this, s, {
                        get: ()=>t[s],
                        set() {}
                    });
                Object.defineProperty(this, "label", {
                    get: ()=>i,
                    set(e) {
                        e !== i && (i = e,
                        this.trigger("labelchange"))
                    }
                })
            }
        }
        const ui = function(e) {
            const t = ["protocol", "hostname", "port", "pathname", "search", "hash", "host"]
              , i = a().createElement("a");
            i.href = e;
            const s = {};
            for (let n = 0; n < t.length; n++)
                s[t[n]] = i[t[n]];
            return "http:" === s.protocol && (s.host = s.host.replace(/:80$/, "")),
            "https:" === s.protocol && (s.host = s.host.replace(/:443$/, "")),
            s.protocol || (s.protocol = n().location.protocol),
            s.host || (s.host = n().location.host),
            s
        }
          , ci = function(e) {
            if (!e.match(/^https?:\/\//)) {
                const t = a().createElement("a");
                t.href = e,
                e = t.href
            }
            return e
        }
          , pi = function(e) {
            if ("string" === typeof e) {
                const t = /^(\/?)([\s\S]*?)((?:\.{1,2}|[^\/]+?)(\.([^\.\/\?]+)))(?:[\/]*|[\?].*)$/.exec(e);
                if (t)
                    return t.pop().toLowerCase()
            }
            return ""
        }
          , mi = function(e, t=n().location) {
            const i = ui(e);
            return (":" === i.protocol ? t.protocol : i.protocol) + i.host !== t.protocol + t.host
        };
        var gi = Object.freeze({
            __proto__: null,
            parseUrl: ui,
            getAbsoluteURL: ci,
            getFileExtension: pi,
            isCrossOrigin: mi
        });
        const fi = function(e, t) {
            const i = new (n().WebVTT.Parser)(n(),n().vttjs,n().WebVTT.StringDecoder())
              , s = [];
            i.oncue = function(e) {
                t.addCue(e)
            }
            ,
            i.onparsingerror = function(e) {
                s.push(e)
            }
            ,
            i.onflush = function() {
                t.trigger({
                    type: "loadeddata",
                    target: t
                })
            }
            ,
            i.parse(e),
            s.length > 0 && (n().console && n().console.groupCollapsed && n().console.groupCollapsed(`Text Track parsing errors for ${t.src}`),
            s.forEach((e=>U.error(e))),
            n().console && n().console.groupEnd && n().console.groupEnd()),
            i.flush()
        }
          , _i = function(e, t) {
            const i = {
                uri: e
            }
              , s = mi(e);
            s && (i.cors = s);
            const r = "use-credentials" === t.tech_.crossOrigin();
            r && (i.withCredentials = r),
            c()(i, gt(this, (function(e, i, s) {
                if (e)
                    return U.error(e, i);
                t.loaded_ = !0,
                "function" !== typeof n().WebVTT ? t.tech_ && t.tech_.any(["vttjsloaded", "vttjserror"], (e=>{
                    if ("vttjserror" !== e.type)
                        return fi(s, t);
                    U.error(`vttjs failed to load, stopping trying to process ${t.src}`)
                }
                )) : fi(s, t)
            }
            )))
        };
        class yi extends di {
            constructor(e={}) {
                if (!e.tech)
                    throw new Error("A tech was not provided.");
                const t = V(e, {
                    kind: li[e.kind] || "subtitles",
                    language: e.language || e.srclang || ""
                });
                let i = hi[t.mode] || "disabled";
                const s = t.default;
                "metadata" !== t.kind && "chapters" !== t.kind || (i = "hidden"),
                super(t),
                this.tech_ = t.tech,
                this.cues_ = [],
                this.activeCues_ = [],
                this.preload_ = !1 !== this.tech_.preloadTextTracks;
                const n = new ri(this.cues_)
                  , r = new ri(this.activeCues_);
                let a = !1;
                this.timeupdateHandler = gt(this, (function(e={}) {
                    this.tech_.isDisposed() || (this.tech_.isReady_ ? (this.activeCues = this.activeCues,
                    a && (this.trigger("cuechange"),
                    a = !1),
                    "timeupdate" !== e.type && (this.rvf_ = this.tech_.requestVideoFrameCallback(this.timeupdateHandler))) : "timeupdate" !== e.type && (this.rvf_ = this.tech_.requestVideoFrameCallback(this.timeupdateHandler)))
                }
                ));
                this.tech_.one("dispose", (()=>{
                    this.stopTracking()
                }
                )),
                "disabled" !== i && this.startTracking(),
                Object.defineProperties(this, {
                    default: {
                        get: ()=>s,
                        set() {}
                    },
                    mode: {
                        get: ()=>i,
                        set(e) {
                            hi[e] && i !== e && (i = e,
                            this.preload_ || "disabled" === i || 0 !== this.cues.length || _i(this.src, this),
                            this.stopTracking(),
                            "disabled" !== i && this.startTracking(),
                            this.trigger("modechange"))
                        }
                    },
                    cues: {
                        get() {
                            return this.loaded_ ? n : null
                        },
                        set() {}
                    },
                    activeCues: {
                        get() {
                            if (!this.loaded_)
                                return null;
                            if (0 === this.cues.length)
                                return r;
                            const e = this.tech_.currentTime()
                              , t = [];
                            for (let i = 0, s = this.cues.length; i < s; i++) {
                                const s = this.cues[i];
                                s.startTime <= e && s.endTime >= e && t.push(s)
                            }
                            if (a = !1,
                            t.length !== this.activeCues_.length)
                                a = !0;
                            else
                                for (let i = 0; i < t.length; i++)
                                    -1 === this.activeCues_.indexOf(t[i]) && (a = !0);
                            return this.activeCues_ = t,
                            r.setCues_(this.activeCues_),
                            r
                        },
                        set() {}
                    }
                }),
                t.src ? (this.src = t.src,
                this.preload_ || (this.loaded_ = !0),
                (this.preload_ || "subtitles" !== t.kind && "captions" !== t.kind) && _i(this.src, this)) : this.loaded_ = !0
            }
            startTracking() {
                this.rvf_ = this.tech_.requestVideoFrameCallback(this.timeupdateHandler),
                this.tech_.on("timeupdate", this.timeupdateHandler)
            }
            stopTracking() {
                this.rvf_ && (this.tech_.cancelVideoFrameCallback(this.rvf_),
                this.rvf_ = void 0),
                this.tech_.off("timeupdate", this.timeupdateHandler)
            }
            addCue(e) {
                let t = e;
                if (!("getCueAsHTML"in t)) {
                    t = new (n().vttjs.VTTCue)(e.startTime,e.endTime,e.text);
                    for (const i in e)
                        i in t || (t[i] = e[i]);
                    t.id = e.id,
                    t.originalCue_ = e
                }
                const i = this.tech_.textTracks();
                for (let s = 0; s < i.length; s++)
                    i[s] !== this && i[s].removeCue(t);
                this.cues_.push(t),
                this.cues.setCues_(this.cues_)
            }
            removeCue(e) {
                let t = this.cues_.length;
                for (; t--; ) {
                    const i = this.cues_[t];
                    if (i === e || i.originalCue_ && i.originalCue_ === e) {
                        this.cues_.splice(t, 1),
                        this.cues.setCues_(this.cues_);
                        break
                    }
                }
            }
        }
        yi.prototype.allowedEvents_ = {
            cuechange: "cuechange"
        };
        class vi extends di {
            constructor(e={}) {
                const t = V(e, {
                    kind: oi[e.kind] || ""
                });
                super(t);
                let i = !1;
                Object.defineProperty(this, "enabled", {
                    get: ()=>i,
                    set(e) {
                        "boolean" === typeof e && e !== i && (i = e,
                        this.trigger("enabledchange"))
                    }
                }),
                t.enabled && (this.enabled = t.enabled),
                this.loaded_ = !0
            }
        }
        class Ti extends di {
            constructor(e={}) {
                const t = V(e, {
                    kind: ai[e.kind] || ""
                });
                super(t);
                let i = !1;
                Object.defineProperty(this, "selected", {
                    get: ()=>i,
                    set(e) {
                        "boolean" === typeof e && e !== i && (i = e,
                        this.trigger("selectedchange"))
                    }
                }),
                t.selected && (this.selected = t.selected)
            }
        }
        class bi extends Tt {
            constructor(e={}) {
                let t;
                super();
                const i = new yi(e);
                this.kind = i.kind,
                this.src = i.src,
                this.srclang = i.language,
                this.label = i.label,
                this.default = i.default,
                Object.defineProperties(this, {
                    readyState: {
                        get: ()=>t
                    },
                    track: {
                        get: ()=>i
                    }
                }),
                t = bi.NONE,
                i.addEventListener("loadeddata", (()=>{
                    t = bi.LOADED,
                    this.trigger({
                        type: "load",
                        target: this
                    })
                }
                ))
            }
        }
        bi.prototype.allowedEvents_ = {
            load: "load"
        },
        bi.NONE = 0,
        bi.LOADING = 1,
        bi.LOADED = 2,
        bi.ERROR = 3;
        const Si = {
            audio: {
                ListClass: class extends ti {
                    constructor(e=[]) {
                        for (let t = e.length - 1; t >= 0; t--)
                            if (e[t].enabled) {
                                ii(e, e[t]);
                                break
                            }
                        super(e),
                        this.changing_ = !1
                    }
                    addTrack(e) {
                        e.enabled && ii(this, e),
                        super.addTrack(e),
                        e.addEventListener && (e.enabledChange_ = ()=>{
                            this.changing_ || (this.changing_ = !0,
                            ii(this, e),
                            this.changing_ = !1,
                            this.trigger("change"))
                        }
                        ,
                        e.addEventListener("enabledchange", e.enabledChange_))
                    }
                    removeTrack(e) {
                        super.removeTrack(e),
                        e.removeEventListener && e.enabledChange_ && (e.removeEventListener("enabledchange", e.enabledChange_),
                        e.enabledChange_ = null)
                    }
                }
                ,
                TrackClass: vi,
                capitalName: "Audio"
            },
            video: {
                ListClass: class extends ti {
                    constructor(e=[]) {
                        for (let t = e.length - 1; t >= 0; t--)
                            if (e[t].selected) {
                                si(e, e[t]);
                                break
                            }
                        super(e),
                        this.changing_ = !1,
                        Object.defineProperty(this, "selectedIndex", {
                            get() {
                                for (let e = 0; e < this.length; e++)
                                    if (this[e].selected)
                                        return e;
                                return -1
                            },
                            set() {}
                        })
                    }
                    addTrack(e) {
                        e.selected && si(this, e),
                        super.addTrack(e),
                        e.addEventListener && (e.selectedChange_ = ()=>{
                            this.changing_ || (this.changing_ = !0,
                            si(this, e),
                            this.changing_ = !1,
                            this.trigger("change"))
                        }
                        ,
                        e.addEventListener("selectedchange", e.selectedChange_))
                    }
                    removeTrack(e) {
                        super.removeTrack(e),
                        e.removeEventListener && e.selectedChange_ && (e.removeEventListener("selectedchange", e.selectedChange_),
                        e.selectedChange_ = null)
                    }
                }
                ,
                TrackClass: Ti,
                capitalName: "Video"
            },
            text: {
                ListClass: ni,
                TrackClass: yi,
                capitalName: "Text"
            }
        };
        Object.keys(Si).forEach((function(e) {
            Si[e].getterName = `${e}Tracks`,
            Si[e].privateName = `${e}Tracks_`
        }
        ));
        const ki = {
            remoteText: {
                ListClass: ni,
                TrackClass: yi,
                capitalName: "RemoteText",
                getterName: "remoteTextTracks",
                privateName: "remoteTextTracks_"
            },
            remoteTextEl: {
                ListClass: class {
                    constructor(e=[]) {
                        this.trackElements_ = [],
                        Object.defineProperty(this, "length", {
                            get() {
                                return this.trackElements_.length
                            }
                        });
                        for (let t = 0, i = e.length; t < i; t++)
                            this.addTrackElement_(e[t])
                    }
                    addTrackElement_(e) {
                        const t = this.trackElements_.length;
                        "" + t in this || Object.defineProperty(this, t, {
                            get() {
                                return this.trackElements_[t]
                            }
                        }),
                        -1 === this.trackElements_.indexOf(e) && this.trackElements_.push(e)
                    }
                    getTrackElementByTrack_(e) {
                        let t;
                        for (let i = 0, s = this.trackElements_.length; i < s; i++)
                            if (e === this.trackElements_[i].track) {
                                t = this.trackElements_[i];
                                break
                            }
                        return t
                    }
                    removeTrackElement_(e) {
                        for (let t = 0, i = this.trackElements_.length; t < i; t++)
                            if (e === this.trackElements_[t]) {
                                this.trackElements_[t].track && "function" === typeof this.trackElements_[t].track.off && this.trackElements_[t].track.off(),
                                "function" === typeof this.trackElements_[t].off && this.trackElements_[t].off(),
                                this.trackElements_.splice(t, 1);
                                break
                            }
                    }
                }
                ,
                TrackClass: bi,
                capitalName: "RemoteTextTrackEls",
                getterName: "remoteTextTrackEls",
                privateName: "remoteTextTrackEls_"
            }
        }
          , Ci = Object.assign({}, Si, ki);
        ki.names = Object.keys(ki),
        Si.names = Object.keys(Si),
        Ci.names = [].concat(ki.names).concat(Si.names);
        class Ei extends Bt {
            constructor(e={}, t=function() {}
            ) {
                e.reportTouchActivity = !1,
                super(null, e, t),
                this.onDurationChange_ = e=>this.onDurationChange(e),
                this.trackProgress_ = e=>this.trackProgress(e),
                this.trackCurrentTime_ = e=>this.trackCurrentTime(e),
                this.stopTrackingCurrentTime_ = e=>this.stopTrackingCurrentTime(e),
                this.disposeSourceHandler_ = e=>this.disposeSourceHandler(e),
                this.queuedHanders_ = new Set,
                this.hasStarted_ = !1,
                this.on("playing", (function() {
                    this.hasStarted_ = !0
                }
                )),
                this.on("loadstart", (function() {
                    this.hasStarted_ = !1
                }
                )),
                Ci.names.forEach((t=>{
                    const i = Ci[t];
                    e && e[i.getterName] && (this[i.privateName] = e[i.getterName])
                }
                )),
                this.featuresProgressEvents || this.manualProgressOn(),
                this.featuresTimeupdateEvents || this.manualTimeUpdatesOn(),
                ["Text", "Audio", "Video"].forEach((t=>{
                    !1 === e[`native${t}Tracks`] && (this[`featuresNative${t}Tracks`] = !1)
                }
                )),
                !1 === e.nativeCaptions || !1 === e.nativeTextTracks ? this.featuresNativeTextTracks = !1 : !0 !== e.nativeCaptions && !0 !== e.nativeTextTracks || (this.featuresNativeTextTracks = !0),
                this.featuresNativeTextTracks || this.emulateTextTracks(),
                this.preloadTextTracks = !1 !== e.preloadTextTracks,
                this.autoRemoteTextTracks_ = new Ci.text.ListClass,
                this.initTrackListeners(),
                e.nativeControlsForTouch || this.emitTapEvents(),
                this.constructor && (this.name_ = this.constructor.name || "Unknown Tech")
            }
            triggerSourceset(e) {
                this.isReady_ || this.one("ready", (()=>this.setTimeout((()=>this.triggerSourceset(e)), 1))),
                this.trigger({
                    src: e,
                    type: "sourceset"
                })
            }
            manualProgressOn() {
                this.on("durationchange", this.onDurationChange_),
                this.manualProgress = !0,
                this.one("ready", this.trackProgress_)
            }
            manualProgressOff() {
                this.manualProgress = !1,
                this.stopTrackingProgress(),
                this.off("durationchange", this.onDurationChange_)
            }
            trackProgress(e) {
                this.stopTrackingProgress(),
                this.progressInterval = this.setInterval(gt(this, (function() {
                    const e = this.bufferedPercent();
                    this.bufferedPercent_ !== e && this.trigger("progress"),
                    this.bufferedPercent_ = e,
                    1 === e && this.stopTrackingProgress()
                }
                )), 500)
            }
            onDurationChange(e) {
                this.duration_ = this.duration()
            }
            buffered() {
                return jt(0, 0)
            }
            bufferedPercent() {
                return Gt(this.buffered(), this.duration_)
            }
            stopTrackingProgress() {
                this.clearInterval(this.progressInterval)
            }
            manualTimeUpdatesOn() {
                this.manualTimeUpdates = !0,
                this.on("play", this.trackCurrentTime_),
                this.on("pause", this.stopTrackingCurrentTime_)
            }
            manualTimeUpdatesOff() {
                this.manualTimeUpdates = !1,
                this.stopTrackingCurrentTime(),
                this.off("play", this.trackCurrentTime_),
                this.off("pause", this.stopTrackingCurrentTime_)
            }
            trackCurrentTime() {
                this.currentTimeInterval && this.stopTrackingCurrentTime(),
                this.currentTimeInterval = this.setInterval((function() {
                    this.trigger({
                        type: "timeupdate",
                        target: this,
                        manuallyTriggered: !0
                    })
                }
                ), 250)
            }
            stopTrackingCurrentTime() {
                this.clearInterval(this.currentTimeInterval),
                this.trigger({
                    type: "timeupdate",
                    target: this,
                    manuallyTriggered: !0
                })
            }
            dispose() {
                this.clearTracks(Si.names),
                this.manualProgress && this.manualProgressOff(),
                this.manualTimeUpdates && this.manualTimeUpdatesOff(),
                super.dispose()
            }
            clearTracks(e) {
                (e = [].concat(e)).forEach((e=>{
                    const t = this[`${e}Tracks`]() || [];
                    let i = t.length;
                    for (; i--; ) {
                        const s = t[i];
                        "text" === e && this.removeRemoteTextTrack(s),
                        t.removeTrack(s)
                    }
                }
                ))
            }
            cleanupAutoTextTracks() {
                const e = this.autoRemoteTextTracks_ || [];
                let t = e.length;
                for (; t--; ) {
                    const i = e[t];
                    this.removeRemoteTextTrack(i)
                }
            }
            reset() {}
            crossOrigin() {}
            setCrossOrigin() {}
            error(e) {
                return void 0 !== e && (this.error_ = new Kt(e),
                this.trigger("error")),
                this.error_
            }
            played() {
                return this.hasStarted_ ? jt(0, 0) : jt()
            }
            play() {}
            setScrubbing(e) {}
            scrubbing() {}
            setCurrentTime(e) {
                this.manualTimeUpdates && this.trigger({
                    type: "timeupdate",
                    target: this,
                    manuallyTriggered: !0
                })
            }
            initTrackListeners() {
                Si.names.forEach((e=>{
                    const t = Si[e]
                      , i = ()=>{
                        this.trigger(`${e}trackchange`)
                    }
                      , s = this[t.getterName]();
                    s.addEventListener("removetrack", i),
                    s.addEventListener("addtrack", i),
                    this.on("dispose", (()=>{
                        s.removeEventListener("removetrack", i),
                        s.removeEventListener("addtrack", i)
                    }
                    ))
                }
                ))
            }
            addWebVttScript_() {
                if (!n().WebVTT)
                    if (a().body.contains(this.el())) {
                        if (!this.options_["vtt.js"] && H(m()) && Object.keys(m()).length > 0)
                            return void this.trigger("vttjsloaded");
                        const e = a().createElement("script");
                        e.src = this.options_["vtt.js"] || "https://vjs.zencdn.net/vttjs/0.14.1/vtt.min.js",
                        e.onload = ()=>{
                            this.trigger("vttjsloaded")
                        }
                        ,
                        e.onerror = ()=>{
                            this.trigger("vttjserror")
                        }
                        ,
                        this.on("dispose", (()=>{
                            e.onload = null,
                            e.onerror = null
                        }
                        )),
                        n().WebVTT = !0,
                        this.el().parentNode.appendChild(e)
                    } else
                        this.ready(this.addWebVttScript_)
            }
            emulateTextTracks() {
                const e = this.textTracks()
                  , t = this.remoteTextTracks()
                  , i = t=>e.addTrack(t.track)
                  , s = t=>e.removeTrack(t.track);
                t.on("addtrack", i),
                t.on("removetrack", s),
                this.addWebVttScript_();
                const n = ()=>this.trigger("texttrackchange")
                  , r = ()=>{
                    n();
                    for (let t = 0; t < e.length; t++) {
                        const i = e[t];
                        i.removeEventListener("cuechange", n),
                        "showing" === i.mode && i.addEventListener("cuechange", n)
                    }
                }
                ;
                r(),
                e.addEventListener("change", r),
                e.addEventListener("addtrack", r),
                e.addEventListener("removetrack", r),
                this.on("dispose", (function() {
                    t.off("addtrack", i),
                    t.off("removetrack", s),
                    e.removeEventListener("change", r),
                    e.removeEventListener("addtrack", r),
                    e.removeEventListener("removetrack", r);
                    for (let t = 0; t < e.length; t++) {
                        e[t].removeEventListener("cuechange", n)
                    }
                }
                ))
            }
            addTextTrack(e, t, i) {
                if (!e)
                    throw new Error("TextTrack kind is required but was not provided");
                return function(e, t, i, s, n={}) {
                    const r = e.textTracks();
                    n.kind = t,
                    i && (n.label = i),
                    s && (n.language = s),
                    n.tech = e;
                    const a = new Ci.text.TrackClass(n);
                    return r.addTrack(a),
                    a
                }(this, e, t, i)
            }
            createRemoteTextTrack(e) {
                const t = V(e, {
                    tech: this
                });
                return new ki.remoteTextEl.TrackClass(t)
            }
            addRemoteTextTrack(e={}, t) {
                const i = this.createRemoteTextTrack(e);
                return "boolean" !== typeof t && (t = !1),
                this.remoteTextTrackEls().addTrackElement_(i),
                this.remoteTextTracks().addTrack(i.track),
                !1 === t && this.ready((()=>this.autoRemoteTextTracks_.addTrack(i.track))),
                i
            }
            removeRemoteTextTrack(e) {
                const t = this.remoteTextTrackEls().getTrackElementByTrack_(e);
                this.remoteTextTrackEls().removeTrackElement_(t),
                this.remoteTextTracks().removeTrack(e),
                this.autoRemoteTextTracks_.removeTrack(e)
            }
            getVideoPlaybackQuality() {
                return {}
            }
            requestPictureInPicture() {
                return Promise.reject()
            }
            disablePictureInPicture() {
                return !0
            }
            setDisablePictureInPicture() {}
            requestVideoFrameCallback(e) {
                const t = st();
                return !this.isReady_ || this.paused() ? (this.queuedHanders_.add(t),
                this.one("playing", (()=>{
                    this.queuedHanders_.has(t) && (this.queuedHanders_.delete(t),
                    e())
                }
                ))) : this.requestNamedAnimationFrame(t, e),
                t
            }
            cancelVideoFrameCallback(e) {
                this.queuedHanders_.has(e) ? this.queuedHanders_.delete(e) : this.cancelNamedAnimationFrame(e)
            }
            setPoster() {}
            playsinline() {}
            setPlaysinline() {}
            overrideNativeAudioTracks(e) {}
            overrideNativeVideoTracks(e) {}
            canPlayType(e) {
                return ""
            }
            static canPlayType(e) {
                return ""
            }
            static canPlaySource(e, t) {
                return Ei.canPlayType(e.type)
            }
            static isTech(e) {
                return e.prototype instanceof Ei || e instanceof Ei || e === Ei
            }
            static registerTech(e, t) {
                if (Ei.techs_ || (Ei.techs_ = {}),
                !Ei.isTech(t))
                    throw new Error(`Tech ${e} must be a Tech`);
                if (!Ei.canPlayType)
                    throw new Error("Techs must have a static canPlayType method on them");
                if (!Ei.canPlaySource)
                    throw new Error("Techs must have a static canPlaySource method on them");
                return e = Mt(e),
                Ei.techs_[e] = t,
                Ei.techs_[Ot(e)] = t,
                "Tech" !== e && Ei.defaultTechOrder_.push(e),
                t
            }
            static getTech(e) {
                if (e)
                    return Ei.techs_ && Ei.techs_[e] ? Ei.techs_[e] : (e = Mt(e),
                    n() && n().videojs && n().videojs[e] ? (U.warn(`The ${e} tech was added to the videojs object when it should be registered using videojs.registerTech(name, tech)`),
                    n().videojs[e]) : void 0)
            }
        }
        Ci.names.forEach((function(e) {
            const t = Ci[e];
            Ei.prototype[t.getterName] = function() {
                return this[t.privateName] = this[t.privateName] || new t.ListClass,
                this[t.privateName]
            }
        }
        )),
        Ei.prototype.featuresVolumeControl = !0,
        Ei.prototype.featuresMuteControl = !0,
        Ei.prototype.featuresFullscreenResize = !1,
        Ei.prototype.featuresPlaybackRate = !1,
        Ei.prototype.featuresProgressEvents = !1,
        Ei.prototype.featuresSourceset = !1,
        Ei.prototype.featuresTimeupdateEvents = !1,
        Ei.prototype.featuresNativeTextTracks = !1,
        Ei.prototype.featuresVideoFrameCallback = !1,
        Ei.withSourceHandlers = function(e) {
            e.registerSourceHandler = function(t, i) {
                let s = e.sourceHandlers;
                s || (s = e.sourceHandlers = []),
                void 0 === i && (i = s.length),
                s.splice(i, 0, t)
            }
            ,
            e.canPlayType = function(t) {
                const i = e.sourceHandlers || [];
                let s;
                for (let e = 0; e < i.length; e++)
                    if (s = i[e].canPlayType(t),
                    s)
                        return s;
                return ""
            }
            ,
            e.selectSourceHandler = function(t, i) {
                const s = e.sourceHandlers || [];
                let n;
                for (let e = 0; e < s.length; e++)
                    if (n = s[e].canHandleSource(t, i),
                    n)
                        return s[e];
                return null
            }
            ,
            e.canPlaySource = function(t, i) {
                const s = e.selectSourceHandler(t, i);
                return s ? s.canHandleSource(t, i) : ""
            }
            ;
            ["seekable", "seeking", "duration"].forEach((function(e) {
                const t = this[e];
                "function" === typeof t && (this[e] = function() {
                    return this.sourceHandler_ && this.sourceHandler_[e] ? this.sourceHandler_[e].apply(this.sourceHandler_, arguments) : t.apply(this, arguments)
                }
                )
            }
            ), e.prototype),
            e.prototype.setSource = function(t) {
                let i = e.selectSourceHandler(t, this.options_);
                i || (e.nativeSourceHandler ? i = e.nativeSourceHandler : U.error("No source handler found for the current source.")),
                this.disposeSourceHandler(),
                this.off("dispose", this.disposeSourceHandler_),
                i !== e.nativeSourceHandler && (this.currentSource_ = t),
                this.sourceHandler_ = i.handleSource(t, this, this.options_),
                this.one("dispose", this.disposeSourceHandler_)
            }
            ,
            e.prototype.disposeSourceHandler = function() {
                this.currentSource_ && (this.clearTracks(["audio", "video"]),
                this.currentSource_ = null),
                this.cleanupAutoTextTracks(),
                this.sourceHandler_ && (this.sourceHandler_.dispose && this.sourceHandler_.dispose(),
                this.sourceHandler_ = null)
            }
        }
        ,
        Bt.registerComponent("Tech", Ei),
        Ei.registerTech("Tech", Ei),
        Ei.defaultTechOrder_ = [];
        const wi = {}
          , xi = {}
          , Ii = {};
        function Pi(e, t, i) {
            e.setTimeout((()=>Ri(t, wi[t.type], i, e)), 1)
        }
        function Ai(e, t, i, s=null) {
            const n = "call" + Mt(i)
              , r = e.reduce(Mi(n), s)
              , a = r === Ii
              , o = a ? null : t[i](r);
            return function(e, t, i, s) {
                for (let n = e.length - 1; n >= 0; n--) {
                    const r = e[n];
                    r[t] && r[t](s, i)
                }
            }(e, i, o, a),
            o
        }
        const Li = {
            buffered: 1,
            currentTime: 1,
            duration: 1,
            muted: 1,
            played: 1,
            paused: 1,
            seekable: 1,
            volume: 1,
            ended: 1
        }
          , Di = {
            setCurrentTime: 1,
            setMuted: 1,
            setVolume: 1
        }
          , Oi = {
            play: 1,
            pause: 1
        };
        function Mi(e) {
            return (t,i)=>t === Ii ? Ii : i[e] ? i[e](t) : t
        }
        function Ri(e={}, t=[], i, s, n=[], r=!1) {
            const [a,...o] = t;
            if ("string" === typeof a)
                Ri(e, wi[a], i, s, n, r);
            else if (a) {
                const t = function(e, t) {
                    const i = xi[e.id()];
                    let s = null;
                    if (void 0 === i || null === i)
                        return s = t(e),
                        xi[e.id()] = [[t, s]],
                        s;
                    for (let n = 0; n < i.length; n++) {
                        const [e,r] = i[n];
                        e === t && (s = r)
                    }
                    return null === s && (s = t(e),
                    i.push([t, s])),
                    s
                }(s, a);
                if (!t.setSource)
                    return n.push(t),
                    Ri(e, o, i, s, n, r);
                t.setSource(Object.assign({}, e), (function(a, l) {
                    if (a)
                        return Ri(e, o, i, s, n, r);
                    n.push(t),
                    Ri(l, e.type === l.type ? o : wi[l.type], i, s, n, r)
                }
                ))
            } else
                o.length ? Ri(e, o, i, s, n, r) : r ? i(e, n) : Ri(e, wi["*"], i, s, n, !0)
        }
        const Ui = {
            opus: "video/ogg",
            ogv: "video/ogg",
            mp4: "video/mp4",
            mov: "video/mp4",
            m4v: "video/mp4",
            mkv: "video/x-matroska",
            m4a: "audio/mp4",
            mp3: "audio/mpeg",
            aac: "audio/aac",
            caf: "audio/x-caf",
            flac: "audio/flac",
            oga: "audio/ogg",
            wav: "audio/wav",
            m3u8: "application/x-mpegURL",
            mpd: "application/dash+xml",
            jpg: "image/jpeg",
            jpeg: "image/jpeg",
            gif: "image/gif",
            png: "image/png",
            svg: "image/svg+xml",
            webp: "image/webp"
        }
          , Bi = function(e="") {
            const t = pi(e);
            return Ui[t.toLowerCase()] || ""
        }
          , Ni = function(e) {
            if (Array.isArray(e)) {
                let t = [];
                e.forEach((function(e) {
                    e = Ni(e),
                    Array.isArray(e) ? t = t.concat(e) : q(e) && t.push(e)
                }
                )),
                e = t
            } else
                e = "string" === typeof e && e.trim() ? [Fi({
                    src: e
                })] : q(e) && "string" === typeof e.src && e.src && e.src.trim() ? [Fi(e)] : [];
            return e
        };
        function Fi(e) {
            if (!e.type) {
                const t = Bi(e.src);
                t && (e.type = t)
            }
            return e
        }
        Bt.registerComponent("MediaLoader", class extends Bt {
            constructor(e, t, i) {
                if (super(e, V({
                    createEl: !1
                }, t), i),
                t.playerOptions.sources && 0 !== t.playerOptions.sources.length)
                    e.src(t.playerOptions.sources);
                else
                    for (let s = 0, n = t.playerOptions.techOrder; s < n.length; s++) {
                        const t = Mt(n[s]);
                        let i = Ei.getTech(t);
                        if (t || (i = Bt.getComponent(t)),
                        i && i.isSupported()) {
                            e.loadTech_(t);
                            break
                        }
                    }
            }
        }
        );
        class ji extends Bt {
            constructor(e, t) {
                super(e, t),
                this.options_.controlText && this.controlText(this.options_.controlText),
                this.handleMouseOver_ = e=>this.handleMouseOver(e),
                this.handleMouseOut_ = e=>this.handleMouseOut(e),
                this.handleClick_ = e=>this.handleClick(e),
                this.handleKeyDown_ = e=>this.handleKeyDown(e),
                this.emitTapEvents(),
                this.enable()
            }
            createEl(e="div", t={}, i={}) {
                t = Object.assign({
                    className: this.buildCSSClass(),
                    tabIndex: 0
                }, t),
                "button" === e && U.error(`Creating a ClickableComponent with an HTML element of ${e} is not supported; use a Button instead.`),
                i = Object.assign({
                    role: "button"
                }, i),
                this.tabIndex_ = t.tabIndex;
                const s = ve(e, t, i);
                return this.player_.options_.experimentalSvgIcons || s.appendChild(ve("span", {
                    className: "vjs-icon-placeholder"
                }, {
                    "aria-hidden": !0
                })),
                this.createControlTextEl(s),
                s
            }
            dispose() {
                this.controlTextEl_ = null,
                super.dispose()
            }
            createControlTextEl(e) {
                return this.controlTextEl_ = ve("span", {
                    className: "vjs-control-text"
                }, {
                    "aria-live": "polite"
                }),
                e && e.appendChild(this.controlTextEl_),
                this.controlText(this.controlText_, e),
                this.controlTextEl_
            }
            controlText(e, t=this.el()) {
                if (void 0 === e)
                    return this.controlText_ || "Need Text";
                const i = this.localize(e);
                this.controlText_ = e,
                Te(this.controlTextEl_, i),
                this.nonIconControl || this.player_.options_.noUITitleAttributes || t.setAttribute("title", i)
            }
            buildCSSClass() {
                return `vjs-control vjs-button ${super.buildCSSClass()}`
            }
            enable() {
                this.enabled_ || (this.enabled_ = !0,
                this.removeClass("vjs-disabled"),
                this.el_.setAttribute("aria-disabled", "false"),
                "undefined" !== typeof this.tabIndex_ && this.el_.setAttribute("tabIndex", this.tabIndex_),
                this.on(["tap", "click"], this.handleClick_),
                this.on("keydown", this.handleKeyDown_))
            }
            disable() {
                this.enabled_ = !1,
                this.addClass("vjs-disabled"),
                this.el_.setAttribute("aria-disabled", "true"),
                "undefined" !== typeof this.tabIndex_ && this.el_.removeAttribute("tabIndex"),
                this.off("mouseover", this.handleMouseOver_),
                this.off("mouseout", this.handleMouseOut_),
                this.off(["tap", "click"], this.handleClick_),
                this.off("keydown", this.handleKeyDown_)
            }
            handleLanguagechange() {
                this.controlText(this.controlText_)
            }
            handleClick(e) {
                this.options_.clickHandler && this.options_.clickHandler.call(this, arguments)
            }
            handleKeyDown(e) {
                l().isEventKey(e, "Space") || l().isEventKey(e, "Enter") ? (e.preventDefault(),
                e.stopPropagation(),
                this.trigger("click")) : super.handleKeyDown(e)
            }
        }
        Bt.registerComponent("ClickableComponent", ji);
        class $i extends ji {
            constructor(e, t) {
                super(e, t),
                this.update(),
                this.update_ = e=>this.update(e),
                e.on("posterchange", this.update_)
            }
            dispose() {
                this.player().off("posterchange", this.update_),
                super.dispose()
            }
            createEl() {
                return ve("div", {
                    className: "vjs-poster"
                })
            }
            crossOrigin(e) {
                if ("undefined" === typeof e)
                    return this.$("img") ? this.$("img").crossOrigin : this.player_.tech_ && this.player_.tech_.isReady_ ? this.player_.crossOrigin() : this.player_.options_.crossOrigin || this.player_.options_.crossorigin || null;
                null === e || "anonymous" === e || "use-credentials" === e ? this.$("img") && (this.$("img").crossOrigin = e) : this.player_.log.warn(`crossOrigin must be null,  "anonymous" or "use-credentials", given "${e}"`)
            }
            update(e) {
                const t = this.player().poster();
                this.setSrc(t),
                t ? this.show() : this.hide()
            }
            setSrc(e) {
                e ? (this.$("img") || this.el_.appendChild(ve("picture", {
                    className: "vjs-poster",
                    tabIndex: -1
                }, {}, ve("img", {
                    loading: "lazy",
                    crossOrigin: this.crossOrigin()
                }, {
                    alt: ""
                }))),
                this.$("img").src = e) : this.el_.textContent = ""
            }
            handleClick(e) {
                this.player_.controls() && (this.player_.tech(!0) && this.player_.tech(!0).focus(),
                this.player_.paused() ? Xt(this.player_.play()) : this.player_.pause())
            }
        }
        $i.prototype.crossorigin = $i.prototype.crossOrigin,
        Bt.registerComponent("PosterImage", $i);
        const qi = {
            monospace: "monospace",
            sansSerif: "sans-serif",
            serif: "serif",
            monospaceSansSerif: '"Andale Mono", "Lucida Console", monospace',
            monospaceSerif: '"Courier New", monospace',
            proportionalSansSerif: "sans-serif",
            proportionalSerif: "serif",
            casual: '"Comic Sans MS", Impact, fantasy',
            script: '"Monotype Corsiva", cursive',
            smallcaps: '"Andale Mono", "Lucida Console", monospace, sans-serif'
        };
        function Hi(e, t) {
            let i;
            if (4 === e.length)
                i = e[1] + e[1] + e[2] + e[2] + e[3] + e[3];
            else {
                if (7 !== e.length)
                    throw new Error("Invalid color code provided, " + e + "; must be formatted as e.g. #f0e or #f604e2.");
                i = e.slice(1)
            }
            return "rgba(" + parseInt(i.slice(0, 2), 16) + "," + parseInt(i.slice(2, 4), 16) + "," + parseInt(i.slice(4, 6), 16) + "," + t + ")"
        }
        function Vi(e, t, i) {
            try {
                e.style[t] = i
            } catch (s) {
                return
            }
        }
        function zi(e) {
            return e ? `${e}px` : ""
        }
        Bt.registerComponent("TextTrackDisplay", class extends Bt {
            constructor(e, t, i) {
                super(e, t, i);
                const s = e=>{
                    this.updateDisplayOverlay(),
                    this.updateDisplay(e)
                }
                ;
                e.on("loadstart", (e=>this.toggleDisplay(e))),
                e.on("texttrackchange", (e=>this.updateDisplay(e))),
                e.on("loadedmetadata", (e=>{
                    this.updateDisplayOverlay(),
                    this.preselectTrack(e)
                }
                )),
                e.ready(gt(this, (function() {
                    if (e.tech_ && e.tech_.featuresNativeTextTracks)
                        return void this.hide();
                    e.on("fullscreenchange", s),
                    e.on("playerresize", s);
                    const t = n().screen.orientation || n()
                      , i = n().screen.orientation ? "change" : "orientationchange";
                    t.addEventListener(i, s),
                    e.on("dispose", (()=>t.removeEventListener(i, s)));
                    const r = this.options_.playerOptions.tracks || [];
                    for (let e = 0; e < r.length; e++)
                        this.player_.addRemoteTextTrack(r[e], !0);
                    this.preselectTrack()
                }
                )))
            }
            preselectTrack() {
                const e = {
                    captions: 1,
                    subtitles: 1
                }
                  , t = this.player_.textTracks()
                  , i = this.player_.cache_.selectedLanguage;
                let s, n, r;
                for (let a = 0; a < t.length; a++) {
                    const o = t[a];
                    i && i.enabled && i.language && i.language === o.language && o.kind in e ? o.kind === i.kind ? r = o : r || (r = o) : i && !i.enabled ? (r = null,
                    s = null,
                    n = null) : o.default && ("descriptions" !== o.kind || s ? o.kind in e && !n && (n = o) : s = o)
                }
                r ? r.mode = "showing" : n ? n.mode = "showing" : s && (s.mode = "showing")
            }
            toggleDisplay() {
                this.player_.tech_ && this.player_.tech_.featuresNativeTextTracks ? this.hide() : this.show()
            }
            createEl() {
                return super.createEl("div", {
                    className: "vjs-text-track-display"
                }, {
                    translate: "yes",
                    "aria-live": "off",
                    "aria-atomic": "true"
                })
            }
            clearDisplay() {
                "function" === typeof n().WebVTT && n().WebVTT.processCues(n(), [], this.el_)
            }
            updateDisplay() {
                const e = this.player_.textTracks()
                  , t = this.options_.allowMultipleShowingTracks;
                if (this.clearDisplay(),
                t) {
                    const t = [];
                    for (let i = 0; i < e.length; ++i) {
                        const s = e[i];
                        "showing" === s.mode && t.push(s)
                    }
                    return void this.updateForTrack(t)
                }
                let i = null
                  , s = null
                  , n = e.length;
                for (; n--; ) {
                    const t = e[n];
                    "showing" === t.mode && ("descriptions" === t.kind ? i = t : s = t)
                }
                s ? ("off" !== this.getAttribute("aria-live") && this.setAttribute("aria-live", "off"),
                this.updateForTrack(s)) : i && ("assertive" !== this.getAttribute("aria-live") && this.setAttribute("aria-live", "assertive"),
                this.updateForTrack(i))
            }
            updateDisplayOverlay() {
                if (!this.player_.videoHeight() || !n().CSS.supports("inset-inline: 10px"))
                    return;
                const e = this.player_.currentWidth()
                  , t = this.player_.currentHeight()
                  , i = e / t
                  , s = this.player_.videoWidth() / this.player_.videoHeight();
                let r = 0
                  , a = 0;
                Math.abs(i - s) > .1 && (i > s ? r = Math.round((e - t * s) / 2) : a = Math.round((t - e / s) / 2)),
                Vi(this.el_, "insetInline", zi(r)),
                Vi(this.el_, "insetBlock", zi(a))
            }
            updateDisplayState(e) {
                const t = this.player_.textTrackSettings.getValues()
                  , i = e.activeCues;
                let s = i.length;
                for (; s--; ) {
                    const e = i[s];
                    if (!e)
                        continue;
                    const r = e.displayState;
                    if (t.color && (r.firstChild.style.color = t.color),
                    t.textOpacity && Vi(r.firstChild, "color", Hi(t.color || "#fff", t.textOpacity)),
                    t.backgroundColor && (r.firstChild.style.backgroundColor = t.backgroundColor),
                    t.backgroundOpacity && Vi(r.firstChild, "backgroundColor", Hi(t.backgroundColor || "#000", t.backgroundOpacity)),
                    t.windowColor && (t.windowOpacity ? Vi(r, "backgroundColor", Hi(t.windowColor, t.windowOpacity)) : r.style.backgroundColor = t.windowColor),
                    t.edgeStyle && ("dropshadow" === t.edgeStyle ? r.firstChild.style.textShadow = "2px 2px 3px #222, 2px 2px 4px #222, 2px 2px 5px #222" : "raised" === t.edgeStyle ? r.firstChild.style.textShadow = "1px 1px #222, 2px 2px #222, 3px 3px #222" : "depressed" === t.edgeStyle ? r.firstChild.style.textShadow = "1px 1px #ccc, 0 1px #ccc, -1px -1px #222, 0 -1px #222" : "uniform" === t.edgeStyle && (r.firstChild.style.textShadow = "0 0 4px #222, 0 0 4px #222, 0 0 4px #222, 0 0 4px #222")),
                    t.fontPercent && 1 !== t.fontPercent) {
                        const e = n().parseFloat(r.style.fontSize);
                        r.style.fontSize = e * t.fontPercent + "px",
                        r.style.height = "auto",
                        r.style.top = "auto"
                    }
                    t.fontFamily && "default" !== t.fontFamily && ("small-caps" === t.fontFamily ? r.firstChild.style.fontVariant = "small-caps" : r.firstChild.style.fontFamily = qi[t.fontFamily])
                }
            }
            updateForTrack(e) {
                if (Array.isArray(e) || (e = [e]),
                "function" !== typeof n().WebVTT || e.every((e=>!e.activeCues)))
                    return;
                const t = [];
                for (let i = 0; i < e.length; ++i) {
                    const s = e[i];
                    for (let e = 0; e < s.activeCues.length; ++e)
                        t.push(s.activeCues[e])
                }
                n().WebVTT.processCues(n(), t, this.el_);
                for (let i = 0; i < e.length; ++i) {
                    const t = e[i];
                    for (let e = 0; e < t.activeCues.length; ++e) {
                        const s = t.activeCues[e].displayState;
                        ke(s, "vjs-text-track-cue", "vjs-text-track-cue-" + (t.language ? t.language : i)),
                        t.language && Pe(s, "lang", t.language)
                    }
                    this.player_.textTrackSettings && this.updateDisplayState(t)
                }
            }
        }
        );
        Bt.registerComponent("LoadingSpinner", class extends Bt {
            createEl() {
                const e = this.player_.isAudio()
                  , t = this.localize(e ? "Audio Player" : "Video Player")
                  , i = ve("span", {
                    className: "vjs-control-text",
                    textContent: this.localize("{1} is loading.", [t])
                })
                  , s = super.createEl("div", {
                    className: "vjs-loading-spinner",
                    dir: "ltr"
                });
                return s.appendChild(i),
                s
            }
            handleLanguagechange() {
                this.$(".vjs-control-text").textContent = this.localize("{1} is loading.", [this.player_.isAudio() ? "Audio Player" : "Video Player"])
            }
        }
        );
        class Wi extends ji {
            createEl(e, t={}, i={}) {
                const s = ve("button", t = Object.assign({
                    className: this.buildCSSClass()
                }, t), i = Object.assign({
                    type: "button"
                }, i));
                return this.player_.options_.experimentalSvgIcons || s.appendChild(ve("span", {
                    className: "vjs-icon-placeholder"
                }, {
                    "aria-hidden": !0
                })),
                this.createControlTextEl(s),
                s
            }
            addChild(e, t={}) {
                const i = this.constructor.name;
                return U.warn(`Adding an actionable (user controllable) child to a Button (${i}) is not supported; use a ClickableComponent instead.`),
                Bt.prototype.addChild.call(this, e, t)
            }
            enable() {
                super.enable(),
                this.el_.removeAttribute("disabled")
            }
            disable() {
                super.disable(),
                this.el_.setAttribute("disabled", "disabled")
            }
            handleKeyDown(e) {
                l().isEventKey(e, "Space") || l().isEventKey(e, "Enter") ? e.stopPropagation() : super.handleKeyDown(e)
            }
        }
        Bt.registerComponent("Button", Wi);
        class Gi extends Wi {
            constructor(e, t) {
                super(e, t),
                this.mouseused_ = !1,
                this.setIcon("play"),
                this.on("mousedown", (e=>this.handleMouseDown(e)))
            }
            buildCSSClass() {
                return "vjs-big-play-button"
            }
            handleClick(e) {
                const t = this.player_.play();
                if (this.mouseused_ && "clientX"in e && "clientY"in e)
                    return Xt(t),
                    void (this.player_.tech(!0) && this.player_.tech(!0).focus());
                const i = this.player_.getChild("controlBar")
                  , s = i && i.getChild("playToggle");
                if (!s)
                    return void this.player_.tech(!0).focus();
                const n = ()=>s.focus();
                Qt(t) ? t.then(n, (()=>{}
                )) : this.setTimeout(n, 1)
            }
            handleKeyDown(e) {
                this.mouseused_ = !1,
                super.handleKeyDown(e)
            }
            handleMouseDown(e) {
                this.mouseused_ = !0
            }
        }
        Gi.prototype.controlText_ = "Play Video",
        Bt.registerComponent("BigPlayButton", Gi);
        Bt.registerComponent("CloseButton", class extends Wi {
            constructor(e, t) {
                super(e, t),
                this.setIcon("cancel"),
                this.controlText(t && t.controlText || this.localize("Close"))
            }
            buildCSSClass() {
                return `vjs-close-button ${super.buildCSSClass()}`
            }
            handleClick(e) {
                this.trigger({
                    type: "close",
                    bubbles: !1
                })
            }
            handleKeyDown(e) {
                l().isEventKey(e, "Esc") ? (e.preventDefault(),
                e.stopPropagation(),
                this.trigger("click")) : super.handleKeyDown(e)
            }
        }
        );
        class Ki extends Wi {
            constructor(e, t={}) {
                super(e, t),
                t.replay = void 0 === t.replay || t.replay,
                this.setIcon("play"),
                this.on(e, "play", (e=>this.handlePlay(e))),
                this.on(e, "pause", (e=>this.handlePause(e))),
                t.replay && this.on(e, "ended", (e=>this.handleEnded(e)))
            }
            buildCSSClass() {
                return `vjs-play-control ${super.buildCSSClass()}`
            }
            handleClick(e) {
                this.player_.paused() ? Xt(this.player_.play()) : this.player_.pause()
            }
            handleSeeked(e) {
                this.removeClass("vjs-ended"),
                this.player_.paused() ? this.handlePause(e) : this.handlePlay(e)
            }
            handlePlay(e) {
                this.removeClass("vjs-ended", "vjs-paused"),
                this.addClass("vjs-playing"),
                this.setIcon("pause"),
                this.controlText("Pause")
            }
            handlePause(e) {
                this.removeClass("vjs-playing"),
                this.addClass("vjs-paused"),
                this.setIcon("play"),
                this.controlText("Play")
            }
            handleEnded(e) {
                this.removeClass("vjs-playing"),
                this.addClass("vjs-ended"),
                this.setIcon("replay"),
                this.controlText("Replay"),
                this.one(this.player_, "seeked", (e=>this.handleSeeked(e)))
            }
        }
        Ki.prototype.controlText_ = "Play",
        Bt.registerComponent("PlayToggle", Ki);
        class Qi extends Bt {
            constructor(e, t) {
                super(e, t),
                this.on(e, ["timeupdate", "ended", "seeking"], (e=>this.update(e))),
                this.updateTextNode_()
            }
            createEl() {
                const e = this.buildCSSClass()
                  , t = super.createEl("div", {
                    className: `${e} vjs-time-control vjs-control`
                })
                  , i = ve("span", {
                    className: "vjs-control-text",
                    textContent: `${this.localize(this.labelText_)}\xa0`
                }, {
                    role: "presentation"
                });
                return t.appendChild(i),
                this.contentEl_ = ve("span", {
                    className: `${e}-display`
                }, {
                    role: "presentation"
                }),
                t.appendChild(this.contentEl_),
                t
            }
            dispose() {
                this.contentEl_ = null,
                this.textNode_ = null,
                super.dispose()
            }
            update(e) {
                (this.player_.options_.enableSmoothSeeking || "seeking" !== e.type) && this.updateContent(e)
            }
            updateTextNode_(e=0) {
                e = zt(e),
                this.formattedTime_ !== e && (this.formattedTime_ = e,
                this.requestNamedAnimationFrame("TimeDisplay#updateTextNode_", (()=>{
                    if (!this.contentEl_)
                        return;
                    let e = this.textNode_;
                    e && this.contentEl_.firstChild !== e && (e = null,
                    U.warn("TimeDisplay#updateTextnode_: Prevented replacement of text node element since it was no longer a child of this node. Appending a new node instead.")),
                    this.textNode_ = a().createTextNode(this.formattedTime_),
                    this.textNode_ && (e ? this.contentEl_.replaceChild(this.textNode_, e) : this.contentEl_.appendChild(this.textNode_))
                }
                )))
            }
            updateContent(e) {}
        }
        Qi.prototype.labelText_ = "Time",
        Qi.prototype.controlText_ = "Time",
        Bt.registerComponent("TimeDisplay", Qi);
        class Xi extends Qi {
            buildCSSClass() {
                return "vjs-current-time"
            }
            updateContent(e) {
                let t;
                t = this.player_.ended() ? this.player_.duration() : this.player_.scrubbing() ? this.player_.getCache().currentTime : this.player_.currentTime(),
                this.updateTextNode_(t)
            }
        }
        Xi.prototype.labelText_ = "Current Time",
        Xi.prototype.controlText_ = "Current Time",
        Bt.registerComponent("CurrentTimeDisplay", Xi);
        class Yi extends Qi {
            constructor(e, t) {
                super(e, t);
                const i = e=>this.updateContent(e);
                this.on(e, "durationchange", i),
                this.on(e, "loadstart", i),
                this.on(e, "loadedmetadata", i)
            }
            buildCSSClass() {
                return "vjs-duration"
            }
            updateContent(e) {
                const t = this.player_.duration();
                this.updateTextNode_(t)
            }
        }
        Yi.prototype.labelText_ = "Duration",
        Yi.prototype.controlText_ = "Duration",
        Bt.registerComponent("DurationDisplay", Yi);
        Bt.registerComponent("TimeDivider", class extends Bt {
            createEl() {
                const e = super.createEl("div", {
                    className: "vjs-time-control vjs-time-divider"
                }, {
                    "aria-hidden": !0
                })
                  , t = super.createEl("div")
                  , i = super.createEl("span", {
                    textContent: "/"
                });
                return t.appendChild(i),
                e.appendChild(t),
                e
            }
        }
        );
        class Ji extends Qi {
            constructor(e, t) {
                super(e, t),
                this.on(e, "durationchange", (e=>this.updateContent(e)))
            }
            buildCSSClass() {
                return "vjs-remaining-time"
            }
            createEl() {
                const e = super.createEl();
                return !1 !== this.options_.displayNegative && e.insertBefore(ve("span", {}, {
                    "aria-hidden": !0
                }, "-"), this.contentEl_),
                e
            }
            updateContent(e) {
                if ("number" !== typeof this.player_.duration())
                    return;
                let t;
                t = this.player_.ended() ? 0 : this.player_.remainingTimeDisplay ? this.player_.remainingTimeDisplay() : this.player_.remainingTime(),
                this.updateTextNode_(t)
            }
        }
        Ji.prototype.labelText_ = "Remaining Time",
        Ji.prototype.controlText_ = "Remaining Time",
        Bt.registerComponent("RemainingTimeDisplay", Ji);
        Bt.registerComponent("LiveDisplay", class extends Bt {
            constructor(e, t) {
                super(e, t),
                this.updateShowing(),
                this.on(this.player(), "durationchange", (e=>this.updateShowing(e)))
            }
            createEl() {
                const e = super.createEl("div", {
                    className: "vjs-live-control vjs-control"
                });
                return this.contentEl_ = ve("div", {
                    className: "vjs-live-display"
                }, {
                    "aria-live": "off"
                }),
                this.contentEl_.appendChild(ve("span", {
                    className: "vjs-control-text",
                    textContent: `${this.localize("Stream Type")}\xa0`
                })),
                this.contentEl_.appendChild(a().createTextNode(this.localize("LIVE"))),
                e.appendChild(this.contentEl_),
                e
            }
            dispose() {
                this.contentEl_ = null,
                super.dispose()
            }
            updateShowing(e) {
                this.player().duration() === 1 / 0 ? this.show() : this.hide()
            }
        }
        );
        class Zi extends Wi {
            constructor(e, t) {
                super(e, t),
                this.updateLiveEdgeStatus(),
                this.player_.liveTracker && (this.updateLiveEdgeStatusHandler_ = e=>this.updateLiveEdgeStatus(e),
                this.on(this.player_.liveTracker, "liveedgechange", this.updateLiveEdgeStatusHandler_))
            }
            createEl() {
                const e = super.createEl("button", {
                    className: "vjs-seek-to-live-control vjs-control"
                });
                return this.setIcon("circle", e),
                this.textEl_ = ve("span", {
                    className: "vjs-seek-to-live-text",
                    textContent: this.localize("LIVE")
                }, {
                    "aria-hidden": "true"
                }),
                e.appendChild(this.textEl_),
                e
            }
            updateLiveEdgeStatus() {
                !this.player_.liveTracker || this.player_.liveTracker.atLiveEdge() ? (this.setAttribute("aria-disabled", !0),
                this.addClass("vjs-at-live-edge"),
                this.controlText("Seek to live, currently playing live")) : (this.setAttribute("aria-disabled", !1),
                this.removeClass("vjs-at-live-edge"),
                this.controlText("Seek to live, currently behind live"))
            }
            handleClick() {
                this.player_.liveTracker.seekToLiveEdge()
            }
            dispose() {
                this.player_.liveTracker && this.off(this.player_.liveTracker, "liveedgechange", this.updateLiveEdgeStatusHandler_),
                this.textEl_ = null,
                super.dispose()
            }
        }
        function es(e, t, i) {
            return e = Number(e),
            Math.min(i, Math.max(t, isNaN(e) ? t : e))
        }
        Zi.prototype.controlText_ = "Seek to live, currently playing live",
        Bt.registerComponent("SeekToLive", Zi);
        var ts = Object.freeze({
            __proto__: null,
            clamp: es
        });
        class is extends Bt {
            constructor(e, t) {
                super(e, t),
                this.handleMouseDown_ = e=>this.handleMouseDown(e),
                this.handleMouseUp_ = e=>this.handleMouseUp(e),
                this.handleKeyDown_ = e=>this.handleKeyDown(e),
                this.handleClick_ = e=>this.handleClick(e),
                this.handleMouseMove_ = e=>this.handleMouseMove(e),
                this.update_ = e=>this.update(e),
                this.bar = this.getChild(this.options_.barName),
                this.vertical(!!this.options_.vertical),
                this.enable()
            }
            enabled() {
                return this.enabled_
            }
            enable() {
                this.enabled() || (this.on("mousedown", this.handleMouseDown_),
                this.on("touchstart", this.handleMouseDown_),
                this.on("keydown", this.handleKeyDown_),
                this.on("click", this.handleClick_),
                this.on(this.player_, "controlsvisible", this.update),
                this.playerEvent && this.on(this.player_, this.playerEvent, this.update),
                this.removeClass("disabled"),
                this.setAttribute("tabindex", 0),
                this.enabled_ = !0)
            }
            disable() {
                if (!this.enabled())
                    return;
                const e = this.bar.el_.ownerDocument;
                this.off("mousedown", this.handleMouseDown_),
                this.off("touchstart", this.handleMouseDown_),
                this.off("keydown", this.handleKeyDown_),
                this.off("click", this.handleClick_),
                this.off(this.player_, "controlsvisible", this.update_),
                this.off(e, "mousemove", this.handleMouseMove_),
                this.off(e, "mouseup", this.handleMouseUp_),
                this.off(e, "touchmove", this.handleMouseMove_),
                this.off(e, "touchend", this.handleMouseUp_),
                this.removeAttribute("tabindex"),
                this.addClass("disabled"),
                this.playerEvent && this.off(this.player_, this.playerEvent, this.update),
                this.enabled_ = !1
            }
            createEl(e, t={}, i={}) {
                return t.className = t.className + " vjs-slider",
                t = Object.assign({
                    tabIndex: 0
                }, t),
                i = Object.assign({
                    role: "slider",
                    "aria-valuenow": 0,
                    "aria-valuemin": 0,
                    "aria-valuemax": 100
                }, i),
                super.createEl(e, t, i)
            }
            handleMouseDown(e) {
                const t = this.bar.el_.ownerDocument;
                "mousedown" === e.type && e.preventDefault(),
                "touchstart" !== e.type || te || e.preventDefault(),
                Le(),
                this.addClass("vjs-sliding"),
                this.trigger("slideractive"),
                this.on(t, "mousemove", this.handleMouseMove_),
                this.on(t, "mouseup", this.handleMouseUp_),
                this.on(t, "touchmove", this.handleMouseMove_),
                this.on(t, "touchend", this.handleMouseUp_),
                this.handleMouseMove(e, !0)
            }
            handleMouseMove(e) {}
            handleMouseUp(e) {
                const t = this.bar.el_.ownerDocument;
                De(),
                this.removeClass("vjs-sliding"),
                this.trigger("sliderinactive"),
                this.off(t, "mousemove", this.handleMouseMove_),
                this.off(t, "mouseup", this.handleMouseUp_),
                this.off(t, "touchmove", this.handleMouseMove_),
                this.off(t, "touchend", this.handleMouseUp_),
                this.update()
            }
            update() {
                if (!this.el_ || !this.bar)
                    return;
                const e = this.getProgress();
                return e === this.progress_ || (this.progress_ = e,
                this.requestNamedAnimationFrame("Slider#update", (()=>{
                    const t = this.vertical() ? "height" : "width";
                    this.bar.el().style[t] = (100 * e).toFixed(2) + "%"
                }
                ))),
                e
            }
            getProgress() {
                return Number(es(this.getPercent(), 0, 1).toFixed(4))
            }
            calculateDistance(e) {
                const t = Re(this.el_, e);
                return this.vertical() ? t.y : t.x
            }
            handleKeyDown(e) {
                l().isEventKey(e, "Left") || l().isEventKey(e, "Down") ? (e.preventDefault(),
                e.stopPropagation(),
                this.stepBack()) : l().isEventKey(e, "Right") || l().isEventKey(e, "Up") ? (e.preventDefault(),
                e.stopPropagation(),
                this.stepForward()) : super.handleKeyDown(e)
            }
            handleClick(e) {
                e.stopPropagation(),
                e.preventDefault()
            }
            vertical(e) {
                if (void 0 === e)
                    return this.vertical_ || !1;
                this.vertical_ = !!e,
                this.vertical_ ? this.addClass("vjs-slider-vertical") : this.addClass("vjs-slider-horizontal")
            }
        }
        Bt.registerComponent("Slider", is);
        const ss = (e,t)=>es(e / t * 100, 0, 100).toFixed(2) + "%";
        Bt.registerComponent("LoadProgressBar", class extends Bt {
            constructor(e, t) {
                super(e, t),
                this.partEls_ = [],
                this.on(e, "progress", (e=>this.update(e)))
            }
            createEl() {
                const e = super.createEl("div", {
                    className: "vjs-load-progress"
                })
                  , t = ve("span", {
                    className: "vjs-control-text"
                })
                  , i = ve("span", {
                    textContent: this.localize("Loaded")
                })
                  , s = a().createTextNode(": ");
                return this.percentageEl_ = ve("span", {
                    className: "vjs-control-text-loaded-percentage",
                    textContent: "0%"
                }),
                e.appendChild(t),
                t.appendChild(i),
                t.appendChild(s),
                t.appendChild(this.percentageEl_),
                e
            }
            dispose() {
                this.partEls_ = null,
                this.percentageEl_ = null,
                super.dispose()
            }
            update(e) {
                this.requestNamedAnimationFrame("LoadProgressBar#update", (()=>{
                    const e = this.player_.liveTracker
                      , t = this.player_.buffered()
                      , i = e && e.isLive() ? e.seekableEnd() : this.player_.duration()
                      , s = this.player_.bufferedEnd()
                      , n = this.partEls_
                      , r = ss(s, i);
                    this.percent_ !== r && (this.el_.style.width = r,
                    Te(this.percentageEl_, r),
                    this.percent_ = r);
                    for (let a = 0; a < t.length; a++) {
                        const e = t.start(a)
                          , i = t.end(a);
                        let r = n[a];
                        r || (r = this.el_.appendChild(ve()),
                        n[a] = r),
                        r.dataset.start === e && r.dataset.end === i || (r.dataset.start = e,
                        r.dataset.end = i,
                        r.style.left = ss(e, s),
                        r.style.width = ss(i - e, s))
                    }
                    for (let a = n.length; a > t.length; a--)
                        this.el_.removeChild(n[a - 1]);
                    n.length = t.length
                }
                ))
            }
        }
        );
        Bt.registerComponent("TimeTooltip", class extends Bt {
            constructor(e, t) {
                super(e, t),
                this.update = ft(gt(this, this.update), mt)
            }
            createEl() {
                return super.createEl("div", {
                    className: "vjs-time-tooltip"
                }, {
                    "aria-hidden": "true"
                })
            }
            update(e, t, i) {
                const s = Me(this.el_)
                  , n = Oe(this.player_.el())
                  , r = e.width * t;
                if (!n || !s)
                    return;
                const a = e.left - n.left + r
                  , o = e.width - r + (n.right - e.right);
                let l = s.width / 2;
                a < l ? l += l - a : o < l && (l = o),
                l < 0 ? l = 0 : l > s.width && (l = s.width),
                l = Math.round(l),
                this.el_.style.right = `-${l}px`,
                this.write(i)
            }
            write(e) {
                Te(this.el_, e)
            }
            updateTime(e, t, i, s) {
                this.requestNamedAnimationFrame("TimeTooltip#updateTime", (()=>{
                    let n;
                    const r = this.player_.duration();
                    if (this.player_.liveTracker && this.player_.liveTracker.isLive()) {
                        const e = this.player_.liveTracker.liveWindow()
                          , i = e - t * e;
                        n = (i < 1 ? "" : "-") + zt(i, e)
                    } else
                        n = zt(i, r);
                    this.update(e, t, n),
                    s && s()
                }
                ))
            }
        }
        );
        class ns extends Bt {
            constructor(e, t) {
                super(e, t),
                this.setIcon("circle"),
                this.update = ft(gt(this, this.update), mt)
            }
            createEl() {
                return super.createEl("div", {
                    className: "vjs-play-progress vjs-slider-bar"
                }, {
                    "aria-hidden": "true"
                })
            }
            update(e, t) {
                const i = this.getChild("timeTooltip");
                if (!i)
                    return;
                const s = this.player_.scrubbing() ? this.player_.getCache().currentTime : this.player_.currentTime();
                i.updateTime(e, t, s)
            }
        }
        ns.prototype.options_ = {
            children: []
        },
        ue || Y || ns.prototype.options_.children.push("timeTooltip"),
        Bt.registerComponent("PlayProgressBar", ns);
        class rs extends Bt {
            constructor(e, t) {
                super(e, t),
                this.update = ft(gt(this, this.update), mt)
            }
            createEl() {
                return super.createEl("div", {
                    className: "vjs-mouse-display"
                })
            }
            update(e, t) {
                const i = t * this.player_.duration();
                this.getChild("timeTooltip").updateTime(e, t, i, (()=>{
                    this.el_.style.left = e.width * t + "px"
                }
                ))
            }
        }
        rs.prototype.options_ = {
            children: ["timeTooltip"]
        },
        Bt.registerComponent("MouseTimeDisplay", rs);
        class as extends is {
            constructor(e, t) {
                super(e, t),
                this.setEventHandlers_()
            }
            setEventHandlers_() {
                this.update_ = gt(this, this.update),
                this.update = ft(this.update_, mt),
                this.on(this.player_, ["ended", "durationchange", "timeupdate"], this.update),
                this.player_.liveTracker && this.on(this.player_.liveTracker, "liveedgechange", this.update),
                this.updateInterval = null,
                this.enableIntervalHandler_ = e=>this.enableInterval_(e),
                this.disableIntervalHandler_ = e=>this.disableInterval_(e),
                this.on(this.player_, ["playing"], this.enableIntervalHandler_),
                this.on(this.player_, ["ended", "pause", "waiting"], this.disableIntervalHandler_),
                "hidden"in a() && "visibilityState"in a() && this.on(a(), "visibilitychange", this.toggleVisibility_)
            }
            toggleVisibility_(e) {
                "hidden" === a().visibilityState ? (this.cancelNamedAnimationFrame("SeekBar#update"),
                this.cancelNamedAnimationFrame("Slider#update"),
                this.disableInterval_(e)) : (this.player_.ended() || this.player_.paused() || this.enableInterval_(),
                this.update())
            }
            enableInterval_() {
                this.updateInterval || (this.updateInterval = this.setInterval(this.update, mt))
            }
            disableInterval_(e) {
                this.player_.liveTracker && this.player_.liveTracker.isLive() && e && "ended" !== e.type || this.updateInterval && (this.clearInterval(this.updateInterval),
                this.updateInterval = null)
            }
            createEl() {
                return super.createEl("div", {
                    className: "vjs-progress-holder"
                }, {
                    "aria-label": this.localize("Progress Bar")
                })
            }
            update(e) {
                if ("hidden" === a().visibilityState)
                    return;
                const t = super.update();
                return this.requestNamedAnimationFrame("SeekBar#update", (()=>{
                    const e = this.player_.ended() ? this.player_.duration() : this.getCurrentTime_()
                      , i = this.player_.liveTracker;
                    let s = this.player_.duration();
                    i && i.isLive() && (s = this.player_.liveTracker.liveCurrentTime()),
                    this.percent_ !== t && (this.el_.setAttribute("aria-valuenow", (100 * t).toFixed(2)),
                    this.percent_ = t),
                    this.currentTime_ === e && this.duration_ === s || (this.el_.setAttribute("aria-valuetext", this.localize("progress bar timing: currentTime={1} duration={2}", [zt(e, s), zt(s, s)], "{1} of {2}")),
                    this.currentTime_ = e,
                    this.duration_ = s),
                    this.bar && this.bar.update(Oe(this.el()), this.getProgress())
                }
                )),
                t
            }
            userSeek_(e) {
                this.player_.liveTracker && this.player_.liveTracker.isLive() && this.player_.liveTracker.nextSeekedFromUser(),
                this.player_.currentTime(e)
            }
            getCurrentTime_() {
                return this.player_.scrubbing() ? this.player_.getCache().currentTime : this.player_.currentTime()
            }
            getPercent() {
                const e = this.getCurrentTime_();
                let t;
                const i = this.player_.liveTracker;
                return i && i.isLive() ? (t = (e - i.seekableStart()) / i.liveWindow(),
                i.atLiveEdge() && (t = 1)) : t = e / this.player_.duration(),
                t
            }
            handleMouseDown(e) {
                $e(e) && (e.stopPropagation(),
                this.videoWasPlaying = !this.player_.paused(),
                this.player_.pause(),
                super.handleMouseDown(e))
            }
            handleMouseMove(e, t=!1) {
                if (!$e(e) || isNaN(this.player_.duration()))
                    return;
                let i;
                t || this.player_.scrubbing() || this.player_.scrubbing(!0);
                const s = this.calculateDistance(e)
                  , n = this.player_.liveTracker;
                if (n && n.isLive()) {
                    if (s >= .99)
                        return void n.seekToLiveEdge();
                    const e = n.seekableStart()
                      , t = n.liveCurrentTime();
                    if (i = e + s * n.liveWindow(),
                    i >= t && (i = t),
                    i <= e && (i = e + .1),
                    i === 1 / 0)
                        return
                } else
                    i = s * this.player_.duration(),
                    i === this.player_.duration() && (i -= .1);
                this.userSeek_(i),
                this.player_.options_.enableSmoothSeeking && this.update()
            }
            enable() {
                super.enable();
                const e = this.getChild("mouseTimeDisplay");
                e && e.show()
            }
            disable() {
                super.disable();
                const e = this.getChild("mouseTimeDisplay");
                e && e.hide()
            }
            handleMouseUp(e) {
                super.handleMouseUp(e),
                e && e.stopPropagation(),
                this.player_.scrubbing(!1),
                this.player_.trigger({
                    type: "timeupdate",
                    target: this,
                    manuallyTriggered: !0
                }),
                this.videoWasPlaying ? Xt(this.player_.play()) : this.update_()
            }
            stepForward() {
                this.userSeek_(this.player_.currentTime() + 5)
            }
            stepBack() {
                this.userSeek_(this.player_.currentTime() - 5)
            }
            handleAction(e) {
                this.player_.paused() ? this.player_.play() : this.player_.pause()
            }
            handleKeyDown(e) {
                const t = this.player_.liveTracker;
                if (l().isEventKey(e, "Space") || l().isEventKey(e, "Enter"))
                    e.preventDefault(),
                    e.stopPropagation(),
                    this.handleAction(e);
                else if (l().isEventKey(e, "Home"))
                    e.preventDefault(),
                    e.stopPropagation(),
                    this.userSeek_(0);
                else if (l().isEventKey(e, "End"))
                    e.preventDefault(),
                    e.stopPropagation(),
                    t && t.isLive() ? this.userSeek_(t.liveCurrentTime()) : this.userSeek_(this.player_.duration());
                else if (/^[0-9]$/.test(l()(e))) {
                    e.preventDefault(),
                    e.stopPropagation();
                    const i = 10 * (l().codes[l()(e)] - l().codes[0]) / 100;
                    t && t.isLive() ? this.userSeek_(t.seekableStart() + t.liveWindow() * i) : this.userSeek_(this.player_.duration() * i)
                } else
                    l().isEventKey(e, "PgDn") ? (e.preventDefault(),
                    e.stopPropagation(),
                    this.userSeek_(this.player_.currentTime() - 60)) : l().isEventKey(e, "PgUp") ? (e.preventDefault(),
                    e.stopPropagation(),
                    this.userSeek_(this.player_.currentTime() + 60)) : super.handleKeyDown(e)
            }
            dispose() {
                this.disableInterval_(),
                this.off(this.player_, ["ended", "durationchange", "timeupdate"], this.update),
                this.player_.liveTracker && this.off(this.player_.liveTracker, "liveedgechange", this.update),
                this.off(this.player_, ["playing"], this.enableIntervalHandler_),
                this.off(this.player_, ["ended", "pause", "waiting"], this.disableIntervalHandler_),
                "hidden"in a() && "visibilityState"in a() && this.off(a(), "visibilitychange", this.toggleVisibility_),
                super.dispose()
            }
        }
        as.prototype.options_ = {
            children: ["loadProgressBar", "playProgressBar"],
            barName: "playProgressBar"
        },
        ue || Y || as.prototype.options_.children.splice(1, 0, "mouseTimeDisplay"),
        Bt.registerComponent("SeekBar", as);
        class os extends Bt {
            constructor(e, t) {
                super(e, t),
                this.handleMouseMove = ft(gt(this, this.handleMouseMove), mt),
                this.throttledHandleMouseSeek = ft(gt(this, this.handleMouseSeek), mt),
                this.handleMouseUpHandler_ = e=>this.handleMouseUp(e),
                this.handleMouseDownHandler_ = e=>this.handleMouseDown(e),
                this.enable()
            }
            createEl() {
                return super.createEl("div", {
                    className: "vjs-progress-control vjs-control"
                })
            }
            handleMouseMove(e) {
                const t = this.getChild("seekBar");
                if (!t)
                    return;
                const i = t.getChild("playProgressBar")
                  , s = t.getChild("mouseTimeDisplay");
                if (!i && !s)
                    return;
                const n = t.el()
                  , r = Me(n);
                let a = Re(n, e).x;
                a = es(a, 0, 1),
                s && s.update(r, a),
                i && i.update(r, t.getProgress())
            }
            handleMouseSeek(e) {
                const t = this.getChild("seekBar");
                t && t.handleMouseMove(e)
            }
            enabled() {
                return this.enabled_
            }
            disable() {
                if (this.children().forEach((e=>e.disable && e.disable())),
                this.enabled() && (this.off(["mousedown", "touchstart"], this.handleMouseDownHandler_),
                this.off(this.el_, "mousemove", this.handleMouseMove),
                this.removeListenersAddedOnMousedownAndTouchstart(),
                this.addClass("disabled"),
                this.enabled_ = !1,
                this.player_.scrubbing())) {
                    const e = this.getChild("seekBar");
                    this.player_.scrubbing(!1),
                    e.videoWasPlaying && Xt(this.player_.play())
                }
            }
            enable() {
                this.children().forEach((e=>e.enable && e.enable())),
                this.enabled() || (this.on(["mousedown", "touchstart"], this.handleMouseDownHandler_),
                this.on(this.el_, "mousemove", this.handleMouseMove),
                this.removeClass("disabled"),
                this.enabled_ = !0)
            }
            removeListenersAddedOnMousedownAndTouchstart() {
                const e = this.el_.ownerDocument;
                this.off(e, "mousemove", this.throttledHandleMouseSeek),
                this.off(e, "touchmove", this.throttledHandleMouseSeek),
                this.off(e, "mouseup", this.handleMouseUpHandler_),
                this.off(e, "touchend", this.handleMouseUpHandler_)
            }
            handleMouseDown(e) {
                const t = this.el_.ownerDocument
                  , i = this.getChild("seekBar");
                i && i.handleMouseDown(e),
                this.on(t, "mousemove", this.throttledHandleMouseSeek),
                this.on(t, "touchmove", this.throttledHandleMouseSeek),
                this.on(t, "mouseup", this.handleMouseUpHandler_),
                this.on(t, "touchend", this.handleMouseUpHandler_)
            }
            handleMouseUp(e) {
                const t = this.getChild("seekBar");
                t && t.handleMouseUp(e),
                this.removeListenersAddedOnMousedownAndTouchstart()
            }
        }
        os.prototype.options_ = {
            children: ["seekBar"]
        },
        Bt.registerComponent("ProgressControl", os);
        class ls extends Wi {
            constructor(e, t) {
                super(e, t),
                this.setIcon("picture-in-picture-enter"),
                this.on(e, ["enterpictureinpicture", "leavepictureinpicture"], (e=>this.handlePictureInPictureChange(e))),
                this.on(e, ["disablepictureinpicturechanged", "loadedmetadata"], (e=>this.handlePictureInPictureEnabledChange(e))),
                this.on(e, ["loadedmetadata", "audioonlymodechange", "audiopostermodechange"], (()=>this.handlePictureInPictureAudioModeChange())),
                this.disable()
            }
            buildCSSClass() {
                return `vjs-picture-in-picture-control vjs-hidden ${super.buildCSSClass()}`
            }
            handlePictureInPictureAudioModeChange() {
                "audio" === this.player_.currentType().substring(0, 5) || this.player_.audioPosterMode() || this.player_.audioOnlyMode() ? (this.player_.isInPictureInPicture() && this.player_.exitPictureInPicture(),
                this.hide()) : this.show()
            }
            handlePictureInPictureEnabledChange() {
                a().pictureInPictureEnabled && !1 === this.player_.disablePictureInPicture() || this.player_.options_.enableDocumentPictureInPicture && "documentPictureInPicture"in n() ? this.enable() : this.disable()
            }
            handlePictureInPictureChange(e) {
                this.player_.isInPictureInPicture() ? (this.setIcon("picture-in-picture-exit"),
                this.controlText("Exit Picture-in-Picture")) : (this.setIcon("picture-in-picture-enter"),
                this.controlText("Picture-in-Picture")),
                this.handlePictureInPictureEnabledChange()
            }
            handleClick(e) {
                this.player_.isInPictureInPicture() ? this.player_.exitPictureInPicture() : this.player_.requestPictureInPicture()
            }
            show() {
                "function" === typeof a().exitPictureInPicture && super.show()
            }
        }
        ls.prototype.controlText_ = "Picture-in-Picture",
        Bt.registerComponent("PictureInPictureToggle", ls);
        class hs extends Wi {
            constructor(e, t) {
                super(e, t),
                this.setIcon("fullscreen-enter"),
                this.on(e, "fullscreenchange", (e=>this.handleFullscreenChange(e))),
                !1 === a()[e.fsApi_.fullscreenEnabled] && this.disable()
            }
            buildCSSClass() {
                return `vjs-fullscreen-control ${super.buildCSSClass()}`
            }
            handleFullscreenChange(e) {
                this.player_.isFullscreen() ? (this.controlText("Exit Fullscreen"),
                this.setIcon("fullscreen-exit")) : (this.controlText("Fullscreen"),
                this.setIcon("fullscreen-enter"))
            }
            handleClick(e) {
                this.player_.isFullscreen() ? this.player_.exitFullscreen() : this.player_.requestFullscreen()
            }
        }
        hs.prototype.controlText_ = "Fullscreen",
        Bt.registerComponent("FullscreenToggle", hs);
        Bt.registerComponent("VolumeLevel", class extends Bt {
            createEl() {
                const e = super.createEl("div", {
                    className: "vjs-volume-level"
                });
                return this.setIcon("circle", e),
                e.appendChild(super.createEl("span", {
                    className: "vjs-control-text"
                })),
                e
            }
        }
        );
        Bt.registerComponent("VolumeLevelTooltip", class extends Bt {
            constructor(e, t) {
                super(e, t),
                this.update = ft(gt(this, this.update), mt)
            }
            createEl() {
                return super.createEl("div", {
                    className: "vjs-volume-tooltip"
                }, {
                    "aria-hidden": "true"
                })
            }
            update(e, t, i, s) {
                if (!i) {
                    const i = Oe(this.el_)
                      , s = Oe(this.player_.el())
                      , n = e.width * t;
                    if (!s || !i)
                        return;
                    const r = e.left - s.left + n
                      , a = e.width - n + (s.right - e.right);
                    let o = i.width / 2;
                    r < o ? o += o - r : a < o && (o = a),
                    o < 0 ? o = 0 : o > i.width && (o = i.width),
                    this.el_.style.right = `-${o}px`
                }
                this.write(`${s}%`)
            }
            write(e) {
                Te(this.el_, e)
            }
            updateVolume(e, t, i, s, n) {
                this.requestNamedAnimationFrame("VolumeLevelTooltip#updateVolume", (()=>{
                    this.update(e, t, i, s.toFixed(0)),
                    n && n()
                }
                ))
            }
        }
        );
        class ds extends Bt {
            constructor(e, t) {
                super(e, t),
                this.update = ft(gt(this, this.update), mt)
            }
            createEl() {
                return super.createEl("div", {
                    className: "vjs-mouse-display"
                })
            }
            update(e, t, i) {
                const s = 100 * t;
                this.getChild("volumeLevelTooltip").updateVolume(e, t, i, s, (()=>{
                    i ? this.el_.style.bottom = e.height * t + "px" : this.el_.style.left = e.width * t + "px"
                }
                ))
            }
        }
        ds.prototype.options_ = {
            children: ["volumeLevelTooltip"]
        },
        Bt.registerComponent("MouseVolumeLevelDisplay", ds);
        class us extends is {
            constructor(e, t) {
                super(e, t),
                this.on("slideractive", (e=>this.updateLastVolume_(e))),
                this.on(e, "volumechange", (e=>this.updateARIAAttributes(e))),
                e.ready((()=>this.updateARIAAttributes()))
            }
            createEl() {
                return super.createEl("div", {
                    className: "vjs-volume-bar vjs-slider-bar"
                }, {
                    "aria-label": this.localize("Volume Level"),
                    "aria-live": "polite"
                })
            }
            handleMouseDown(e) {
                $e(e) && super.handleMouseDown(e)
            }
            handleMouseMove(e) {
                const t = this.getChild("mouseVolumeLevelDisplay");
                if (t) {
                    const i = this.el()
                      , s = Oe(i)
                      , n = this.vertical();
                    let r = Re(i, e);
                    r = n ? r.y : r.x,
                    r = es(r, 0, 1),
                    t.update(s, r, n)
                }
                $e(e) && (this.checkMuted(),
                this.player_.volume(this.calculateDistance(e)))
            }
            checkMuted() {
                this.player_.muted() && this.player_.muted(!1)
            }
            getPercent() {
                return this.player_.muted() ? 0 : this.player_.volume()
            }
            stepForward() {
                this.checkMuted(),
                this.player_.volume(this.player_.volume() + .1)
            }
            stepBack() {
                this.checkMuted(),
                this.player_.volume(this.player_.volume() - .1)
            }
            updateARIAAttributes(e) {
                const t = this.player_.muted() ? 0 : this.volumeAsPercentage_();
                this.el_.setAttribute("aria-valuenow", t),
                this.el_.setAttribute("aria-valuetext", t + "%")
            }
            volumeAsPercentage_() {
                return Math.round(100 * this.player_.volume())
            }
            updateLastVolume_() {
                const e = this.player_.volume();
                this.one("sliderinactive", (()=>{
                    0 === this.player_.volume() && this.player_.lastVolume_(e)
                }
                ))
            }
        }
        us.prototype.options_ = {
            children: ["volumeLevel"],
            barName: "volumeLevel"
        },
        ue || Y || us.prototype.options_.children.splice(0, 0, "mouseVolumeLevelDisplay"),
        us.prototype.playerEvent = "volumechange",
        Bt.registerComponent("VolumeBar", us);
        class cs extends Bt {
            constructor(e, t={}) {
                t.vertical = t.vertical || !1,
                ("undefined" === typeof t.volumeBar || H(t.volumeBar)) && (t.volumeBar = t.volumeBar || {},
                t.volumeBar.vertical = t.vertical),
                super(e, t),
                function(e, t) {
                    t.tech_ && !t.tech_.featuresVolumeControl && e.addClass("vjs-hidden"),
                    e.on(t, "loadstart", (function() {
                        t.tech_.featuresVolumeControl ? e.removeClass("vjs-hidden") : e.addClass("vjs-hidden")
                    }
                    ))
                }(this, e),
                this.throttledHandleMouseMove = ft(gt(this, this.handleMouseMove), mt),
                this.handleMouseUpHandler_ = e=>this.handleMouseUp(e),
                this.on("mousedown", (e=>this.handleMouseDown(e))),
                this.on("touchstart", (e=>this.handleMouseDown(e))),
                this.on("mousemove", (e=>this.handleMouseMove(e))),
                this.on(this.volumeBar, ["focus", "slideractive"], (()=>{
                    this.volumeBar.addClass("vjs-slider-active"),
                    this.addClass("vjs-slider-active"),
                    this.trigger("slideractive")
                }
                )),
                this.on(this.volumeBar, ["blur", "sliderinactive"], (()=>{
                    this.volumeBar.removeClass("vjs-slider-active"),
                    this.removeClass("vjs-slider-active"),
                    this.trigger("sliderinactive")
                }
                ))
            }
            createEl() {
                let e = "vjs-volume-horizontal";
                return this.options_.vertical && (e = "vjs-volume-vertical"),
                super.createEl("div", {
                    className: `vjs-volume-control vjs-control ${e}`
                })
            }
            handleMouseDown(e) {
                const t = this.el_.ownerDocument;
                this.on(t, "mousemove", this.throttledHandleMouseMove),
                this.on(t, "touchmove", this.throttledHandleMouseMove),
                this.on(t, "mouseup", this.handleMouseUpHandler_),
                this.on(t, "touchend", this.handleMouseUpHandler_)
            }
            handleMouseUp(e) {
                const t = this.el_.ownerDocument;
                this.off(t, "mousemove", this.throttledHandleMouseMove),
                this.off(t, "touchmove", this.throttledHandleMouseMove),
                this.off(t, "mouseup", this.handleMouseUpHandler_),
                this.off(t, "touchend", this.handleMouseUpHandler_)
            }
            handleMouseMove(e) {
                this.volumeBar.handleMouseMove(e)
            }
        }
        cs.prototype.options_ = {
            children: ["volumeBar"]
        },
        Bt.registerComponent("VolumeControl", cs);
        class ps extends Wi {
            constructor(e, t) {
                super(e, t),
                function(e, t) {
                    t.tech_ && !t.tech_.featuresMuteControl && e.addClass("vjs-hidden"),
                    e.on(t, "loadstart", (function() {
                        t.tech_.featuresMuteControl ? e.removeClass("vjs-hidden") : e.addClass("vjs-hidden")
                    }
                    ))
                }(this, e),
                this.on(e, ["loadstart", "volumechange"], (e=>this.update(e)))
            }
            buildCSSClass() {
                return `vjs-mute-control ${super.buildCSSClass()}`
            }
            handleClick(e) {
                const t = this.player_.volume()
                  , i = this.player_.lastVolume_();
                if (0 === t) {
                    const e = i < .1 ? .1 : i;
                    this.player_.volume(e),
                    this.player_.muted(!1)
                } else
                    this.player_.muted(!this.player_.muted())
            }
            update(e) {
                this.updateIcon_(),
                this.updateControlText_()
            }
            updateIcon_() {
                const e = this.player_.volume();
                let t = 3;
                this.setIcon("volume-high"),
                ue && this.player_.tech_ && this.player_.tech_.el_ && this.player_.muted(this.player_.tech_.el_.muted),
                0 === e || this.player_.muted() ? (this.setIcon("volume-mute"),
                t = 0) : e < .33 ? (this.setIcon("volume-low"),
                t = 1) : e < .67 && (this.setIcon("volume-medium"),
                t = 2),
                Ce(this.el_, [0, 1, 2, 3].reduce(((e,t)=>e + `${t ? " " : ""}vjs-vol-${t}`), "")),
                ke(this.el_, `vjs-vol-${t}`)
            }
            updateControlText_() {
                const e = this.player_.muted() || 0 === this.player_.volume() ? "Unmute" : "Mute";
                this.controlText() !== e && this.controlText(e)
            }
        }
        ps.prototype.controlText_ = "Mute",
        Bt.registerComponent("MuteToggle", ps);
        class ms extends Bt {
            constructor(e, t={}) {
                "undefined" !== typeof t.inline ? t.inline = t.inline : t.inline = !0,
                ("undefined" === typeof t.volumeControl || H(t.volumeControl)) && (t.volumeControl = t.volumeControl || {},
                t.volumeControl.vertical = !t.inline),
                super(e, t),
                this.handleKeyPressHandler_ = e=>this.handleKeyPress(e),
                this.on(e, ["loadstart"], (e=>this.volumePanelState_(e))),
                this.on(this.muteToggle, "keyup", (e=>this.handleKeyPress(e))),
                this.on(this.volumeControl, "keyup", (e=>this.handleVolumeControlKeyUp(e))),
                this.on("keydown", (e=>this.handleKeyPress(e))),
                this.on("mouseover", (e=>this.handleMouseOver(e))),
                this.on("mouseout", (e=>this.handleMouseOut(e))),
                this.on(this.volumeControl, ["slideractive"], this.sliderActive_),
                this.on(this.volumeControl, ["sliderinactive"], this.sliderInactive_)
            }
            sliderActive_() {
                this.addClass("vjs-slider-active")
            }
            sliderInactive_() {
                this.removeClass("vjs-slider-active")
            }
            volumePanelState_() {
                this.volumeControl.hasClass("vjs-hidden") && this.muteToggle.hasClass("vjs-hidden") && this.addClass("vjs-hidden"),
                this.volumeControl.hasClass("vjs-hidden") && !this.muteToggle.hasClass("vjs-hidden") && this.addClass("vjs-mute-toggle-only")
            }
            createEl() {
                let e = "vjs-volume-panel-horizontal";
                return this.options_.inline || (e = "vjs-volume-panel-vertical"),
                super.createEl("div", {
                    className: `vjs-volume-panel vjs-control ${e}`
                })
            }
            dispose() {
                this.handleMouseOut(),
                super.dispose()
            }
            handleVolumeControlKeyUp(e) {
                l().isEventKey(e, "Esc") && this.muteToggle.focus()
            }
            handleMouseOver(e) {
                this.addClass("vjs-hover"),
                lt(a(), "keyup", this.handleKeyPressHandler_)
            }
            handleMouseOut(e) {
                this.removeClass("vjs-hover"),
                ht(a(), "keyup", this.handleKeyPressHandler_)
            }
            handleKeyPress(e) {
                l().isEventKey(e, "Esc") && this.handleMouseOut()
            }
        }
        ms.prototype.options_ = {
            children: ["muteToggle", "volumeControl"]
        },
        Bt.registerComponent("VolumePanel", ms);
        class gs extends Wi {
            constructor(e, t) {
                super(e, t),
                this.validOptions = [5, 10, 30],
                this.skipTime = this.getSkipForwardTime(),
                this.skipTime && this.validOptions.includes(this.skipTime) ? (this.setIcon(`forward-${this.skipTime}`),
                this.controlText(this.localize("Skip forward {1} seconds", [this.skipTime])),
                this.show()) : this.hide()
            }
            getSkipForwardTime() {
                const e = this.options_.playerOptions;
                return e.controlBar && e.controlBar.skipButtons && e.controlBar.skipButtons.forward
            }
            buildCSSClass() {
                return `vjs-skip-forward-${this.getSkipForwardTime()} ${super.buildCSSClass()}`
            }
            handleClick(e) {
                if (isNaN(this.player_.duration()))
                    return;
                const t = this.player_.currentTime()
                  , i = this.player_.liveTracker
                  , s = i && i.isLive() ? i.seekableEnd() : this.player_.duration();
                let n;
                n = t + this.skipTime <= s ? t + this.skipTime : s,
                this.player_.currentTime(n)
            }
            handleLanguagechange() {
                this.controlText(this.localize("Skip forward {1} seconds", [this.skipTime]))
            }
        }
        gs.prototype.controlText_ = "Skip Forward",
        Bt.registerComponent("SkipForward", gs);
        class fs extends Wi {
            constructor(e, t) {
                super(e, t),
                this.validOptions = [5, 10, 30],
                this.skipTime = this.getSkipBackwardTime(),
                this.skipTime && this.validOptions.includes(this.skipTime) ? (this.setIcon(`replay-${this.skipTime}`),
                this.controlText(this.localize("Skip backward {1} seconds", [this.skipTime])),
                this.show()) : this.hide()
            }
            getSkipBackwardTime() {
                const e = this.options_.playerOptions;
                return e.controlBar && e.controlBar.skipButtons && e.controlBar.skipButtons.backward
            }
            buildCSSClass() {
                return `vjs-skip-backward-${this.getSkipBackwardTime()} ${super.buildCSSClass()}`
            }
            handleClick(e) {
                const t = this.player_.currentTime()
                  , i = this.player_.liveTracker
                  , s = i && i.isLive() && i.seekableStart();
                let n;
                n = s && t - this.skipTime <= s ? s : t >= this.skipTime ? t - this.skipTime : 0,
                this.player_.currentTime(n)
            }
            handleLanguagechange() {
                this.controlText(this.localize("Skip backward {1} seconds", [this.skipTime]))
            }
        }
        fs.prototype.controlText_ = "Skip Backward",
        Bt.registerComponent("SkipBackward", fs);
        class _s extends Bt {
            constructor(e, t) {
                super(e, t),
                t && (this.menuButton_ = t.menuButton),
                this.focusedChild_ = -1,
                this.on("keydown", (e=>this.handleKeyDown(e))),
                this.boundHandleBlur_ = e=>this.handleBlur(e),
                this.boundHandleTapClick_ = e=>this.handleTapClick(e)
            }
            addEventListenerForItem(e) {
                e instanceof Bt && (this.on(e, "blur", this.boundHandleBlur_),
                this.on(e, ["tap", "click"], this.boundHandleTapClick_))
            }
            removeEventListenerForItem(e) {
                e instanceof Bt && (this.off(e, "blur", this.boundHandleBlur_),
                this.off(e, ["tap", "click"], this.boundHandleTapClick_))
            }
            removeChild(e) {
                "string" === typeof e && (e = this.getChild(e)),
                this.removeEventListenerForItem(e),
                super.removeChild(e)
            }
            addItem(e) {
                const t = this.addChild(e);
                t && this.addEventListenerForItem(t)
            }
            createEl() {
                const e = this.options_.contentElType || "ul";
                this.contentEl_ = ve(e, {
                    className: "vjs-menu-content"
                }),
                this.contentEl_.setAttribute("role", "menu");
                const t = super.createEl("div", {
                    append: this.contentEl_,
                    className: "vjs-menu"
                });
                return t.appendChild(this.contentEl_),
                lt(t, "click", (function(e) {
                    e.preventDefault(),
                    e.stopImmediatePropagation()
                }
                )),
                t
            }
            dispose() {
                this.contentEl_ = null,
                this.boundHandleBlur_ = null,
                this.boundHandleTapClick_ = null,
                super.dispose()
            }
            handleBlur(e) {
                const t = e.relatedTarget || a().activeElement;
                if (!this.children().some((e=>e.el() === t))) {
                    const e = this.menuButton_;
                    e && e.buttonPressed_ && t !== e.el().firstChild && e.unpressButton()
                }
            }
            handleTapClick(e) {
                if (this.menuButton_) {
                    this.menuButton_.unpressButton();
                    const t = this.children();
                    if (!Array.isArray(t))
                        return;
                    const i = t.filter((t=>t.el() === e.target))[0];
                    if (!i)
                        return;
                    "CaptionSettingsMenuItem" !== i.name() && this.menuButton_.focus()
                }
            }
            handleKeyDown(e) {
                l().isEventKey(e, "Left") || l().isEventKey(e, "Down") ? (e.preventDefault(),
                e.stopPropagation(),
                this.stepForward()) : (l().isEventKey(e, "Right") || l().isEventKey(e, "Up")) && (e.preventDefault(),
                e.stopPropagation(),
                this.stepBack())
            }
            stepForward() {
                let e = 0;
                void 0 !== this.focusedChild_ && (e = this.focusedChild_ + 1),
                this.focus(e)
            }
            stepBack() {
                let e = 0;
                void 0 !== this.focusedChild_ && (e = this.focusedChild_ - 1),
                this.focus(e)
            }
            focus(e=0) {
                const t = this.children().slice();
                t.length && t[0].hasClass("vjs-menu-title") && t.shift(),
                t.length > 0 && (e < 0 ? e = 0 : e >= t.length && (e = t.length - 1),
                this.focusedChild_ = e,
                t[e].el_.focus())
            }
        }
        Bt.registerComponent("Menu", _s);
        class ys extends Bt {
            constructor(e, t={}) {
                super(e, t),
                this.menuButton_ = new Wi(e,t),
                this.menuButton_.controlText(this.controlText_),
                this.menuButton_.el_.setAttribute("aria-haspopup", "true");
                const i = Wi.prototype.buildCSSClass();
                this.menuButton_.el_.className = this.buildCSSClass() + " " + i,
                this.menuButton_.removeClass("vjs-control"),
                this.addChild(this.menuButton_),
                this.update(),
                this.enabled_ = !0;
                const s = e=>this.handleClick(e);
                this.handleMenuKeyUp_ = e=>this.handleMenuKeyUp(e),
                this.on(this.menuButton_, "tap", s),
                this.on(this.menuButton_, "click", s),
                this.on(this.menuButton_, "keydown", (e=>this.handleKeyDown(e))),
                this.on(this.menuButton_, "mouseenter", (()=>{
                    this.addClass("vjs-hover"),
                    this.menu.show(),
                    lt(a(), "keyup", this.handleMenuKeyUp_)
                }
                )),
                this.on("mouseleave", (e=>this.handleMouseLeave(e))),
                this.on("keydown", (e=>this.handleSubmenuKeyDown(e)))
            }
            update() {
                const e = this.createMenu();
                this.menu && (this.menu.dispose(),
                this.removeChild(this.menu)),
                this.menu = e,
                this.addChild(e),
                this.buttonPressed_ = !1,
                this.menuButton_.el_.setAttribute("aria-expanded", "false"),
                this.items && this.items.length <= this.hideThreshold_ ? (this.hide(),
                this.menu.contentEl_.removeAttribute("role")) : (this.show(),
                this.menu.contentEl_.setAttribute("role", "menu"))
            }
            createMenu() {
                const e = new _s(this.player_,{
                    menuButton: this
                });
                if (this.hideThreshold_ = 0,
                this.options_.title) {
                    const t = ve("li", {
                        className: "vjs-menu-title",
                        textContent: Mt(this.options_.title),
                        tabIndex: -1
                    })
                      , i = new Bt(this.player_,{
                        el: t
                    });
                    e.addItem(i)
                }
                if (this.items = this.createItems(),
                this.items)
                    for (let t = 0; t < this.items.length; t++)
                        e.addItem(this.items[t]);
                return e
            }
            createItems() {}
            createEl() {
                return super.createEl("div", {
                    className: this.buildWrapperCSSClass()
                }, {})
            }
            setIcon(e) {
                super.setIcon(e, this.menuButton_.el_)
            }
            buildWrapperCSSClass() {
                let e = "vjs-menu-button";
                !0 === this.options_.inline ? e += "-inline" : e += "-popup";
                return `vjs-menu-button ${e} ${Wi.prototype.buildCSSClass()} ${super.buildCSSClass()}`
            }
            buildCSSClass() {
                let e = "vjs-menu-button";
                return !0 === this.options_.inline ? e += "-inline" : e += "-popup",
                `vjs-menu-button ${e} ${super.buildCSSClass()}`
            }
            controlText(e, t=this.menuButton_.el()) {
                return this.menuButton_.controlText(e, t)
            }
            dispose() {
                this.handleMouseLeave(),
                super.dispose()
            }
            handleClick(e) {
                this.buttonPressed_ ? this.unpressButton() : this.pressButton()
            }
            handleMouseLeave(e) {
                this.removeClass("vjs-hover"),
                ht(a(), "keyup", this.handleMenuKeyUp_)
            }
            focus() {
                this.menuButton_.focus()
            }
            blur() {
                this.menuButton_.blur()
            }
            handleKeyDown(e) {
                l().isEventKey(e, "Esc") || l().isEventKey(e, "Tab") ? (this.buttonPressed_ && this.unpressButton(),
                l().isEventKey(e, "Tab") || (e.preventDefault(),
                this.menuButton_.focus())) : (l().isEventKey(e, "Up") || l().isEventKey(e, "Down")) && (this.buttonPressed_ || (e.preventDefault(),
                this.pressButton()))
            }
            handleMenuKeyUp(e) {
                (l().isEventKey(e, "Esc") || l().isEventKey(e, "Tab")) && this.removeClass("vjs-hover")
            }
            handleSubmenuKeyPress(e) {
                this.handleSubmenuKeyDown(e)
            }
            handleSubmenuKeyDown(e) {
                (l().isEventKey(e, "Esc") || l().isEventKey(e, "Tab")) && (this.buttonPressed_ && this.unpressButton(),
                l().isEventKey(e, "Tab") || (e.preventDefault(),
                this.menuButton_.focus()))
            }
            pressButton() {
                if (this.enabled_) {
                    if (this.buttonPressed_ = !0,
                    this.menu.show(),
                    this.menu.lockShowing(),
                    this.menuButton_.el_.setAttribute("aria-expanded", "true"),
                    ue && _e())
                        return;
                    this.menu.focus()
                }
            }
            unpressButton() {
                this.enabled_ && (this.buttonPressed_ = !1,
                this.menu.unlockShowing(),
                this.menu.hide(),
                this.menuButton_.el_.setAttribute("aria-expanded", "false"))
            }
            disable() {
                this.unpressButton(),
                this.enabled_ = !1,
                this.addClass("vjs-disabled"),
                this.menuButton_.disable()
            }
            enable() {
                this.enabled_ = !0,
                this.removeClass("vjs-disabled"),
                this.menuButton_.enable()
            }
        }
        Bt.registerComponent("MenuButton", ys);
        class vs extends ys {
            constructor(e, t) {
                const i = t.tracks;
                if (super(e, t),
                this.items.length <= 1 && this.hide(),
                !i)
                    return;
                const s = gt(this, this.update);
                i.addEventListener("removetrack", s),
                i.addEventListener("addtrack", s),
                i.addEventListener("labelchange", s),
                this.player_.on("ready", s),
                this.player_.on("dispose", (function() {
                    i.removeEventListener("removetrack", s),
                    i.removeEventListener("addtrack", s),
                    i.removeEventListener("labelchange", s)
                }
                ))
            }
        }
        Bt.registerComponent("TrackButton", vs);
        const Ts = ["Tab", "Esc", "Up", "Down", "Right", "Left"];
        class bs extends ji {
            constructor(e, t) {
                super(e, t),
                this.selectable = t.selectable,
                this.isSelected_ = t.selected || !1,
                this.multiSelectable = t.multiSelectable,
                this.selected(this.isSelected_),
                this.selectable ? this.multiSelectable ? this.el_.setAttribute("role", "menuitemcheckbox") : this.el_.setAttribute("role", "menuitemradio") : this.el_.setAttribute("role", "menuitem")
            }
            createEl(e, t, i) {
                this.nonIconControl = !0;
                const s = super.createEl("li", Object.assign({
                    className: "vjs-menu-item",
                    tabIndex: -1
                }, t), i)
                  , n = ve("span", {
                    className: "vjs-menu-item-text",
                    textContent: this.localize(this.options_.label)
                });
                return this.player_.options_.experimentalSvgIcons ? s.appendChild(n) : s.replaceChild(n, s.querySelector(".vjs-icon-placeholder")),
                s
            }
            handleKeyDown(e) {
                Ts.some((t=>l().isEventKey(e, t))) || super.handleKeyDown(e)
            }
            handleClick(e) {
                this.selected(!0)
            }
            selected(e) {
                this.selectable && (e ? (this.addClass("vjs-selected"),
                this.el_.setAttribute("aria-checked", "true"),
                this.controlText(", selected"),
                this.isSelected_ = !0) : (this.removeClass("vjs-selected"),
                this.el_.setAttribute("aria-checked", "false"),
                this.controlText(""),
                this.isSelected_ = !1))
            }
        }
        Bt.registerComponent("MenuItem", bs);
        class Ss extends bs {
            constructor(e, t) {
                const i = t.track
                  , s = e.textTracks();
                t.label = i.label || i.language || "Unknown",
                t.selected = "showing" === i.mode,
                super(e, t),
                this.track = i,
                this.kinds = (t.kinds || [t.kind || this.track.kind]).filter(Boolean);
                const r = (...e)=>{
                    this.handleTracksChange.apply(this, e)
                }
                  , o = (...e)=>{
                    this.handleSelectedLanguageChange.apply(this, e)
                }
                ;
                if (e.on(["loadstart", "texttrackchange"], r),
                s.addEventListener("change", r),
                s.addEventListener("selectedlanguagechange", o),
                this.on("dispose", (function() {
                    e.off(["loadstart", "texttrackchange"], r),
                    s.removeEventListener("change", r),
                    s.removeEventListener("selectedlanguagechange", o)
                }
                )),
                void 0 === s.onchange) {
                    let e;
                    this.on(["tap", "click"], (function() {
                        if ("object" !== typeof n().Event)
                            try {
                                e = new (n().Event)("change")
                            } catch (t) {}
                        e || (e = a().createEvent("Event"),
                        e.initEvent("change", !0, !0)),
                        s.dispatchEvent(e)
                    }
                    ))
                }
                this.handleTracksChange()
            }
            handleClick(e) {
                const t = this.track
                  , i = this.player_.textTracks();
                if (super.handleClick(e),
                i)
                    for (let s = 0; s < i.length; s++) {
                        const e = i[s];
                        -1 !== this.kinds.indexOf(e.kind) && (e === t ? "showing" !== e.mode && (e.mode = "showing") : "disabled" !== e.mode && (e.mode = "disabled"))
                    }
            }
            handleTracksChange(e) {
                const t = "showing" === this.track.mode;
                t !== this.isSelected_ && this.selected(t)
            }
            handleSelectedLanguageChange(e) {
                if ("showing" === this.track.mode) {
                    const e = this.player_.cache_.selectedLanguage;
                    if (e && e.enabled && e.language === this.track.language && e.kind !== this.track.kind)
                        return;
                    this.player_.cache_.selectedLanguage = {
                        enabled: !0,
                        language: this.track.language,
                        kind: this.track.kind
                    }
                }
            }
            dispose() {
                this.track = null,
                super.dispose()
            }
        }
        Bt.registerComponent("TextTrackMenuItem", Ss);
        class ks extends Ss {
            constructor(e, t) {
                t.track = {
                    player: e,
                    kind: t.kind,
                    kinds: t.kinds,
                    default: !1,
                    mode: "disabled"
                },
                t.kinds || (t.kinds = [t.kind]),
                t.label ? t.track.label = t.label : t.track.label = t.kinds.join(" and ") + " off",
                t.selectable = !0,
                t.multiSelectable = !1,
                super(e, t)
            }
            handleTracksChange(e) {
                const t = this.player().textTracks();
                let i = !0;
                for (let s = 0, n = t.length; s < n; s++) {
                    const e = t[s];
                    if (this.options_.kinds.indexOf(e.kind) > -1 && "showing" === e.mode) {
                        i = !1;
                        break
                    }
                }
                i !== this.isSelected_ && this.selected(i)
            }
            handleSelectedLanguageChange(e) {
                const t = this.player().textTracks();
                let i = !0;
                for (let s = 0, n = t.length; s < n; s++) {
                    const e = t[s];
                    if (["captions", "descriptions", "subtitles"].indexOf(e.kind) > -1 && "showing" === e.mode) {
                        i = !1;
                        break
                    }
                }
                i && (this.player_.cache_.selectedLanguage = {
                    enabled: !1
                })
            }
            handleLanguagechange() {
                this.$(".vjs-menu-item-text").textContent = this.player_.localize(this.options_.label),
                super.handleLanguagechange()
            }
        }
        Bt.registerComponent("OffTextTrackMenuItem", ks);
        class Cs extends vs {
            constructor(e, t={}) {
                t.tracks = e.textTracks(),
                super(e, t)
            }
            createItems(e=[], t=Ss) {
                let i;
                this.label_ && (i = `${this.label_} off`),
                e.push(new ks(this.player_,{
                    kinds: this.kinds_,
                    kind: this.kind_,
                    label: i
                })),
                this.hideThreshold_ += 1;
                const s = this.player_.textTracks();
                Array.isArray(this.kinds_) || (this.kinds_ = [this.kind_]);
                for (let n = 0; n < s.length; n++) {
                    const i = s[n];
                    if (this.kinds_.indexOf(i.kind) > -1) {
                        const s = new t(this.player_,{
                            track: i,
                            kinds: this.kinds_,
                            kind: this.kind_,
                            selectable: !0,
                            multiSelectable: !1
                        });
                        s.addClass(`vjs-${i.kind}-menu-item`),
                        e.push(s)
                    }
                }
                return e
            }
        }
        Bt.registerComponent("TextTrackButton", Cs);
        class Es extends bs {
            constructor(e, t) {
                const i = t.track
                  , s = t.cue
                  , n = e.currentTime();
                t.selectable = !0,
                t.multiSelectable = !1,
                t.label = s.text,
                t.selected = s.startTime <= n && n < s.endTime,
                super(e, t),
                this.track = i,
                this.cue = s
            }
            handleClick(e) {
                super.handleClick(),
                this.player_.currentTime(this.cue.startTime)
            }
        }
        Bt.registerComponent("ChaptersTrackMenuItem", Es);
        class ws extends Cs {
            constructor(e, t, i) {
                super(e, t, i),
                this.setIcon("chapters"),
                this.selectCurrentItem_ = ()=>{
                    this.items.forEach((e=>{
                        e.selected(this.track_.activeCues[0] === e.cue)
                    }
                    ))
                }
            }
            buildCSSClass() {
                return `vjs-chapters-button ${super.buildCSSClass()}`
            }
            buildWrapperCSSClass() {
                return `vjs-chapters-button ${super.buildWrapperCSSClass()}`
            }
            update(e) {
                if (e && e.track && "chapters" !== e.track.kind)
                    return;
                const t = this.findChaptersTrack();
                t !== this.track_ ? (this.setTrack(t),
                super.update()) : (!this.items || t && t.cues && t.cues.length !== this.items.length) && super.update()
            }
            setTrack(e) {
                if (this.track_ !== e) {
                    if (this.updateHandler_ || (this.updateHandler_ = this.update.bind(this)),
                    this.track_) {
                        const e = this.player_.remoteTextTrackEls().getTrackElementByTrack_(this.track_);
                        e && e.removeEventListener("load", this.updateHandler_),
                        this.track_.removeEventListener("cuechange", this.selectCurrentItem_),
                        this.track_ = null
                    }
                    if (this.track_ = e,
                    this.track_) {
                        this.track_.mode = "hidden";
                        const e = this.player_.remoteTextTrackEls().getTrackElementByTrack_(this.track_);
                        e && e.addEventListener("load", this.updateHandler_),
                        this.track_.addEventListener("cuechange", this.selectCurrentItem_)
                    }
                }
            }
            findChaptersTrack() {
                const e = this.player_.textTracks() || [];
                for (let t = e.length - 1; t >= 0; t--) {
                    const i = e[t];
                    if (i.kind === this.kind_)
                        return i
                }
            }
            getMenuCaption() {
                return this.track_ && this.track_.label ? this.track_.label : this.localize(Mt(this.kind_))
            }
            createMenu() {
                return this.options_.title = this.getMenuCaption(),
                super.createMenu()
            }
            createItems() {
                const e = [];
                if (!this.track_)
                    return e;
                const t = this.track_.cues;
                if (!t)
                    return e;
                for (let i = 0, s = t.length; i < s; i++) {
                    const s = t[i]
                      , n = new Es(this.player_,{
                        track: this.track_,
                        cue: s
                    });
                    e.push(n)
                }
                return e
            }
        }
        ws.prototype.kind_ = "chapters",
        ws.prototype.controlText_ = "Chapters",
        Bt.registerComponent("ChaptersButton", ws);
        class xs extends Cs {
            constructor(e, t, i) {
                super(e, t, i),
                this.setIcon("audio-description");
                const s = e.textTracks()
                  , n = gt(this, this.handleTracksChange);
                s.addEventListener("change", n),
                this.on("dispose", (function() {
                    s.removeEventListener("change", n)
                }
                ))
            }
            handleTracksChange(e) {
                const t = this.player().textTracks();
                let i = !1;
                for (let s = 0, n = t.length; s < n; s++) {
                    const e = t[s];
                    if (e.kind !== this.kind_ && "showing" === e.mode) {
                        i = !0;
                        break
                    }
                }
                i ? this.disable() : this.enable()
            }
            buildCSSClass() {
                return `vjs-descriptions-button ${super.buildCSSClass()}`
            }
            buildWrapperCSSClass() {
                return `vjs-descriptions-button ${super.buildWrapperCSSClass()}`
            }
        }
        xs.prototype.kind_ = "descriptions",
        xs.prototype.controlText_ = "Descriptions",
        Bt.registerComponent("DescriptionsButton", xs);
        class Is extends Cs {
            constructor(e, t, i) {
                super(e, t, i),
                this.setIcon("subtitles")
            }
            buildCSSClass() {
                return `vjs-subtitles-button ${super.buildCSSClass()}`
            }
            buildWrapperCSSClass() {
                return `vjs-subtitles-button ${super.buildWrapperCSSClass()}`
            }
        }
        Is.prototype.kind_ = "subtitles",
        Is.prototype.controlText_ = "Subtitles",
        Bt.registerComponent("SubtitlesButton", Is);
        class Ps extends Ss {
            constructor(e, t) {
                t.track = {
                    player: e,
                    kind: t.kind,
                    label: t.kind + " settings",
                    selectable: !1,
                    default: !1,
                    mode: "disabled"
                },
                t.selectable = !1,
                t.name = "CaptionSettingsMenuItem",
                super(e, t),
                this.addClass("vjs-texttrack-settings"),
                this.controlText(", opens " + t.kind + " settings dialog")
            }
            handleClick(e) {
                this.player().getChild("textTrackSettings").open()
            }
            handleLanguagechange() {
                this.$(".vjs-menu-item-text").textContent = this.player_.localize(this.options_.kind + " settings"),
                super.handleLanguagechange()
            }
        }
        Bt.registerComponent("CaptionSettingsMenuItem", Ps);
        class As extends Cs {
            constructor(e, t, i) {
                super(e, t, i),
                this.setIcon("captions")
            }
            buildCSSClass() {
                return `vjs-captions-button ${super.buildCSSClass()}`
            }
            buildWrapperCSSClass() {
                return `vjs-captions-button ${super.buildWrapperCSSClass()}`
            }
            createItems() {
                const e = [];
                return this.player().tech_ && this.player().tech_.featuresNativeTextTracks || !this.player().getChild("textTrackSettings") || (e.push(new Ps(this.player_,{
                    kind: this.kind_
                })),
                this.hideThreshold_ += 1),
                super.createItems(e)
            }
        }
        As.prototype.kind_ = "captions",
        As.prototype.controlText_ = "Captions",
        Bt.registerComponent("CaptionsButton", As);
        class Ls extends Ss {
            createEl(e, t, i) {
                const s = super.createEl(e, t, i)
                  , n = s.querySelector(".vjs-menu-item-text");
                return "captions" === this.options_.track.kind && (this.player_.options_.experimentalSvgIcons ? this.setIcon("captions", s) : n.appendChild(ve("span", {
                    className: "vjs-icon-placeholder"
                }, {
                    "aria-hidden": !0
                })),
                n.appendChild(ve("span", {
                    className: "vjs-control-text",
                    textContent: ` ${this.localize("Captions")}`
                }))),
                s
            }
        }
        Bt.registerComponent("SubsCapsMenuItem", Ls);
        class Ds extends Cs {
            constructor(e, t={}) {
                super(e, t),
                this.label_ = "subtitles",
                this.setIcon("subtitles"),
                ["en", "en-us", "en-ca", "fr-ca"].indexOf(this.player_.language_) > -1 && (this.label_ = "captions",
                this.setIcon("captions")),
                this.menuButton_.controlText(Mt(this.label_))
            }
            buildCSSClass() {
                return `vjs-subs-caps-button ${super.buildCSSClass()}`
            }
            buildWrapperCSSClass() {
                return `vjs-subs-caps-button ${super.buildWrapperCSSClass()}`
            }
            createItems() {
                let e = [];
                return this.player().tech_ && this.player().tech_.featuresNativeTextTracks || !this.player().getChild("textTrackSettings") || (e.push(new Ps(this.player_,{
                    kind: this.label_
                })),
                this.hideThreshold_ += 1),
                e = super.createItems(e, Ls),
                e
            }
        }
        Ds.prototype.kinds_ = ["captions", "subtitles"],
        Ds.prototype.controlText_ = "Subtitles",
        Bt.registerComponent("SubsCapsButton", Ds);
        class Os extends bs {
            constructor(e, t) {
                const i = t.track
                  , s = e.audioTracks();
                t.label = i.label || i.language || "Unknown",
                t.selected = i.enabled,
                super(e, t),
                this.track = i,
                this.addClass(`vjs-${i.kind}-menu-item`);
                const n = (...e)=>{
                    this.handleTracksChange.apply(this, e)
                }
                ;
                s.addEventListener("change", n),
                this.on("dispose", (()=>{
                    s.removeEventListener("change", n)
                }
                ))
            }
            createEl(e, t, i) {
                const s = super.createEl(e, t, i)
                  , n = s.querySelector(".vjs-menu-item-text");
                return ["main-desc", "description"].indexOf(this.options_.track.kind) >= 0 && (n.appendChild(ve("span", {
                    className: "vjs-icon-placeholder"
                }, {
                    "aria-hidden": !0
                })),
                n.appendChild(ve("span", {
                    className: "vjs-control-text",
                    textContent: " " + this.localize("Descriptions")
                }))),
                s
            }
            handleClick(e) {
                if (super.handleClick(e),
                this.track.enabled = !0,
                this.player_.tech_.featuresNativeAudioTracks) {
                    const e = this.player_.audioTracks();
                    for (let t = 0; t < e.length; t++) {
                        const i = e[t];
                        i !== this.track && (i.enabled = i === this.track)
                    }
                }
            }
            handleTracksChange(e) {
                this.selected(this.track.enabled)
            }
        }
        Bt.registerComponent("AudioTrackMenuItem", Os);
        class Ms extends vs {
            constructor(e, t={}) {
                t.tracks = e.audioTracks(),
                super(e, t),
                this.setIcon("audio")
            }
            buildCSSClass() {
                return `vjs-audio-button ${super.buildCSSClass()}`
            }
            buildWrapperCSSClass() {
                return `vjs-audio-button ${super.buildWrapperCSSClass()}`
            }
            createItems(e=[]) {
                this.hideThreshold_ = 1;
                const t = this.player_.audioTracks();
                for (let i = 0; i < t.length; i++) {
                    const s = t[i];
                    e.push(new Os(this.player_,{
                        track: s,
                        selectable: !0,
                        multiSelectable: !1
                    }))
                }
                return e
            }
        }
        Ms.prototype.controlText_ = "Audio Track",
        Bt.registerComponent("AudioTrackButton", Ms);
        class Rs extends bs {
            constructor(e, t) {
                const i = t.rate
                  , s = parseFloat(i, 10);
                t.label = i,
                t.selected = s === e.playbackRate(),
                t.selectable = !0,
                t.multiSelectable = !1,
                super(e, t),
                this.label = i,
                this.rate = s,
                this.on(e, "ratechange", (e=>this.update(e)))
            }
            handleClick(e) {
                super.handleClick(),
                this.player().playbackRate(this.rate)
            }
            update(e) {
                this.selected(this.player().playbackRate() === this.rate)
            }
        }
        Rs.prototype.contentElType = "button",
        Bt.registerComponent("PlaybackRateMenuItem", Rs);
        class Us extends ys {
            constructor(e, t) {
                super(e, t),
                this.menuButton_.el_.setAttribute("aria-describedby", this.labelElId_),
                this.updateVisibility(),
                this.updateLabel(),
                this.on(e, "loadstart", (e=>this.updateVisibility(e))),
                this.on(e, "ratechange", (e=>this.updateLabel(e))),
                this.on(e, "playbackrateschange", (e=>this.handlePlaybackRateschange(e)))
            }
            createEl() {
                const e = super.createEl();
                return this.labelElId_ = "vjs-playback-rate-value-label-" + this.id_,
                this.labelEl_ = ve("div", {
                    className: "vjs-playback-rate-value",
                    id: this.labelElId_,
                    textContent: "1x"
                }),
                e.appendChild(this.labelEl_),
                e
            }
            dispose() {
                this.labelEl_ = null,
                super.dispose()
            }
            buildCSSClass() {
                return `vjs-playback-rate ${super.buildCSSClass()}`
            }
            buildWrapperCSSClass() {
                return `vjs-playback-rate ${super.buildWrapperCSSClass()}`
            }
            createItems() {
                const e = this.playbackRates()
                  , t = [];
                for (let i = e.length - 1; i >= 0; i--)
                    t.push(new Rs(this.player(),{
                        rate: e[i] + "x"
                    }));
                return t
            }
            handlePlaybackRateschange(e) {
                this.update()
            }
            playbackRates() {
                const e = this.player();
                return e.playbackRates && e.playbackRates() || []
            }
            playbackRateSupported() {
                return this.player().tech_ && this.player().tech_.featuresPlaybackRate && this.playbackRates() && this.playbackRates().length > 0
            }
            updateVisibility(e) {
                this.playbackRateSupported() ? this.removeClass("vjs-hidden") : this.addClass("vjs-hidden")
            }
            updateLabel(e) {
                this.playbackRateSupported() && (this.labelEl_.textContent = this.player().playbackRate() + "x")
            }
        }
        Us.prototype.controlText_ = "Playback Rate",
        Bt.registerComponent("PlaybackRateMenuButton", Us);
        class Bs extends Bt {
            buildCSSClass() {
                return `vjs-spacer ${super.buildCSSClass()}`
            }
            createEl(e="div", t={}, i={}) {
                return t.className || (t.className = this.buildCSSClass()),
                super.createEl(e, t, i)
            }
        }
        Bt.registerComponent("Spacer", Bs);
        Bt.registerComponent("CustomControlSpacer", class extends Bs {
            buildCSSClass() {
                return `vjs-custom-control-spacer ${super.buildCSSClass()}`
            }
            createEl() {
                return super.createEl("div", {
                    className: this.buildCSSClass(),
                    textContent: "\xa0"
                })
            }
        }
        );
        class Ns extends Bt {
            createEl() {
                return super.createEl("div", {
                    className: "vjs-control-bar",
                    dir: "ltr"
                })
            }
        }
        Ns.prototype.options_ = {
            children: ["playToggle", "skipBackward", "skipForward", "volumePanel", "currentTimeDisplay", "timeDivider", "durationDisplay", "progressControl", "liveDisplay", "seekToLive", "remainingTimeDisplay", "customControlSpacer", "playbackRateMenuButton", "chaptersButton", "descriptionsButton", "subsCapsButton", "audioTrackButton", "pictureInPictureToggle", "fullscreenToggle"]
        },
        Bt.registerComponent("ControlBar", Ns);
        class Fs extends ei {
            constructor(e, t) {
                super(e, t),
                this.on(e, "error", (e=>{
                    this.close(),
                    this.open(e)
                }
                ))
            }
            buildCSSClass() {
                return `vjs-error-display ${super.buildCSSClass()}`
            }
            content() {
                const e = this.player().error();
                return e ? this.localize(e.message) : ""
            }
        }
        Fs.prototype.options_ = Object.assign({}, ei.prototype.options_, {
            pauseOnOpen: !1,
            fillAlways: !0,
            temporary: !1,
            uncloseable: !0
        }),
        Bt.registerComponent("ErrorDisplay", Fs);
        const js = "vjs-text-track-settings"
          , $s = ["#000", "Black"]
          , qs = ["#00F", "Blue"]
          , Hs = ["#0FF", "Cyan"]
          , Vs = ["#0F0", "Green"]
          , zs = ["#F0F", "Magenta"]
          , Ws = ["#F00", "Red"]
          , Gs = ["#FFF", "White"]
          , Ks = ["#FF0", "Yellow"]
          , Qs = ["1", "Opaque"]
          , Xs = ["0.5", "Semi-Transparent"]
          , Ys = ["0", "Transparent"]
          , Js = {
            backgroundColor: {
                selector: ".vjs-bg-color > select",
                id: "captions-background-color-%s",
                label: "Color",
                options: [$s, Gs, Ws, Vs, qs, Ks, zs, Hs]
            },
            backgroundOpacity: {
                selector: ".vjs-bg-opacity > select",
                id: "captions-background-opacity-%s",
                label: "Opacity",
                options: [Qs, Xs, Ys]
            },
            color: {
                selector: ".vjs-text-color > select",
                id: "captions-foreground-color-%s",
                label: "Color",
                options: [Gs, $s, Ws, Vs, qs, Ks, zs, Hs]
            },
            edgeStyle: {
                selector: ".vjs-edge-style > select",
                id: "%s",
                label: "Text Edge Style",
                options: [["none", "None"], ["raised", "Raised"], ["depressed", "Depressed"], ["uniform", "Uniform"], ["dropshadow", "Drop shadow"]]
            },
            fontFamily: {
                selector: ".vjs-font-family > select",
                id: "captions-font-family-%s",
                label: "Font Family",
                options: [["proportionalSansSerif", "Proportional Sans-Serif"], ["monospaceSansSerif", "Monospace Sans-Serif"], ["proportionalSerif", "Proportional Serif"], ["monospaceSerif", "Monospace Serif"], ["casual", "Casual"], ["script", "Script"], ["small-caps", "Small Caps"]]
            },
            fontPercent: {
                selector: ".vjs-font-percent > select",
                id: "captions-font-size-%s",
                label: "Font Size",
                options: [["0.50", "50%"], ["0.75", "75%"], ["1.00", "100%"], ["1.25", "125%"], ["1.50", "150%"], ["1.75", "175%"], ["2.00", "200%"], ["3.00", "300%"], ["4.00", "400%"]],
                default: 2,
                parser: e=>"1.00" === e ? null : Number(e)
            },
            textOpacity: {
                selector: ".vjs-text-opacity > select",
                id: "captions-foreground-opacity-%s",
                label: "Opacity",
                options: [Qs, Xs]
            },
            windowColor: {
                selector: ".vjs-window-color > select",
                id: "captions-window-color-%s",
                label: "Color"
            },
            windowOpacity: {
                selector: ".vjs-window-opacity > select",
                id: "captions-window-opacity-%s",
                label: "Opacity",
                options: [Ys, Xs, Qs]
            }
        };
        function Zs(e, t) {
            if (t && (e = t(e)),
            e && "none" !== e)
                return e
        }
        Js.windowColor.options = Js.backgroundColor.options;
        Bt.registerComponent("TextTrackSettings", class extends ei {
            constructor(e, t) {
                t.temporary = !1,
                super(e, t),
                this.updateDisplay = this.updateDisplay.bind(this),
                this.fill(),
                this.hasBeenOpened_ = this.hasBeenFilled_ = !0,
                this.endDialog = ve("p", {
                    className: "vjs-control-text",
                    textContent: this.localize("End of dialog window.")
                }),
                this.el().appendChild(this.endDialog),
                this.setDefaults(),
                void 0 === t.persistTextTrackSettings && (this.options_.persistTextTrackSettings = this.options_.playerOptions.persistTextTrackSettings),
                this.on(this.$(".vjs-done-button"), "click", (()=>{
                    this.saveSettings(),
                    this.close()
                }
                )),
                this.on(this.$(".vjs-default-button"), "click", (()=>{
                    this.setDefaults(),
                    this.updateDisplay()
                }
                )),
                j(Js, (e=>{
                    this.on(this.$(e.selector), "change", this.updateDisplay)
                }
                )),
                this.options_.persistTextTrackSettings && this.restoreSettings()
            }
            dispose() {
                this.endDialog = null,
                super.dispose()
            }
            createElSelect_(e, t="", i="label") {
                const s = Js[e]
                  , n = s.id.replace("%s", this.id_)
                  , r = [t, n].join(" ").trim()
                  , a = `vjs_select_${st()}`;
                return [`<${i} id="${n}"${"label" === i ? ` for="${a}" class="vjs-label"` : ""}>`, this.localize(s.label), `</${i}>`, `<select aria-labelledby="${r}" id="${a}">`].concat(s.options.map((e=>{
                    const t = n + "-" + e[1].replace(/\W+/g, "");
                    return [`<option id="${t}" value="${e[0]}" `, `aria-labelledby="${r} ${t}">`, this.localize(e[1]), "</option>"].join("")
                }
                ))).concat("</select>").join("")
            }
            createElFgColor_() {
                const e = `captions-text-legend-${this.id_}`;
                return ['<fieldset class="vjs-fg vjs-track-setting">', `<legend id="${e}">`, this.localize("Text"), "</legend>", '<span class="vjs-text-color">', this.createElSelect_("color", e), "</span>", '<span class="vjs-text-opacity vjs-opacity">', this.createElSelect_("textOpacity", e), "</span>", "</fieldset>"].join("")
            }
            createElBgColor_() {
                const e = `captions-background-${this.id_}`;
                return ['<fieldset class="vjs-bg vjs-track-setting">', `<legend id="${e}">`, this.localize("Text Background"), "</legend>", '<span class="vjs-bg-color">', this.createElSelect_("backgroundColor", e), "</span>", '<span class="vjs-bg-opacity vjs-opacity">', this.createElSelect_("backgroundOpacity", e), "</span>", "</fieldset>"].join("")
            }
            createElWinColor_() {
                const e = `captions-window-${this.id_}`;
                return ['<fieldset class="vjs-window vjs-track-setting">', `<legend id="${e}">`, this.localize("Caption Area Background"), "</legend>", '<span class="vjs-window-color">', this.createElSelect_("windowColor", e), "</span>", '<span class="vjs-window-opacity vjs-opacity">', this.createElSelect_("windowOpacity", e), "</span>", "</fieldset>"].join("")
            }
            createElColors_() {
                return ve("div", {
                    className: "vjs-track-settings-colors",
                    innerHTML: [this.createElFgColor_(), this.createElBgColor_(), this.createElWinColor_()].join("")
                })
            }
            createElFont_() {
                return ve("div", {
                    className: "vjs-track-settings-font",
                    innerHTML: ['<fieldset class="vjs-font-percent vjs-track-setting">', this.createElSelect_("fontPercent", "", "legend"), "</fieldset>", '<fieldset class="vjs-edge-style vjs-track-setting">', this.createElSelect_("edgeStyle", "", "legend"), "</fieldset>", '<fieldset class="vjs-font-family vjs-track-setting">', this.createElSelect_("fontFamily", "", "legend"), "</fieldset>"].join("")
                })
            }
            createElControls_() {
                const e = this.localize("restore all settings to the default values");
                return ve("div", {
                    className: "vjs-track-settings-controls",
                    innerHTML: [`<button type="button" class="vjs-default-button" title="${e}">`, this.localize("Reset"), `<span class="vjs-control-text"> ${e}</span>`, "</button>", `<button type="button" class="vjs-done-button">${this.localize("Done")}</button>`].join("")
                })
            }
            content() {
                return [this.createElColors_(), this.createElFont_(), this.createElControls_()]
            }
            label() {
                return this.localize("Caption Settings Dialog")
            }
            description() {
                return this.localize("Beginning of dialog window. Escape will cancel and close the window.")
            }
            buildCSSClass() {
                return super.buildCSSClass() + " vjs-text-track-settings"
            }
            getValues() {
                return $(Js, ((e,t,i)=>{
                    const s = (n = this.$(t.selector),
                    r = t.parser,
                    Zs(n.options[n.options.selectedIndex].value, r));
                    var n, r;
                    return void 0 !== s && (e[i] = s),
                    e
                }
                ), {})
            }
            setValues(e) {
                j(Js, ((t,i)=>{
                    !function(e, t, i) {
                        if (t)
                            for (let s = 0; s < e.options.length; s++)
                                if (Zs(e.options[s].value, i) === t) {
                                    e.selectedIndex = s;
                                    break
                                }
                    }(this.$(t.selector), e[i], t.parser)
                }
                ))
            }
            setDefaults() {
                j(Js, (e=>{
                    const t = e.hasOwnProperty("default") ? e.default : 0;
                    this.$(e.selector).selectedIndex = t
                }
                ))
            }
            restoreSettings() {
                let e;
                try {
                    e = JSON.parse(n().localStorage.getItem(js))
                } catch (t) {
                    U.warn(t)
                }
                e && this.setValues(e)
            }
            saveSettings() {
                if (!this.options_.persistTextTrackSettings)
                    return;
                const e = this.getValues();
                try {
                    Object.keys(e).length ? n().localStorage.setItem(js, JSON.stringify(e)) : n().localStorage.removeItem(js)
                } catch (t) {
                    U.warn(t)
                }
            }
            updateDisplay() {
                const e = this.player_.getChild("textTrackDisplay");
                e && e.updateDisplay()
            }
            conditionalBlur_() {
                this.previouslyActiveEl_ = null;
                const e = this.player_.controlBar
                  , t = e && e.subsCapsButton
                  , i = e && e.captionsButton;
                t ? t.focus() : i && i.focus()
            }
            handleLanguagechange() {
                this.fill()
            }
        }
        );
        Bt.registerComponent("ResizeManager", class extends Bt {
            constructor(e, t) {
                let i = t.ResizeObserver || n().ResizeObserver;
                null === t.ResizeObserver && (i = !1);
                super(e, V({
                    createEl: !i,
                    reportTouchActivity: !1
                }, t)),
                this.ResizeObserver = t.ResizeObserver || n().ResizeObserver,
                this.loadListener_ = null,
                this.resizeObserver_ = null,
                this.debouncedHandler_ = _t((()=>{
                    this.resizeHandler()
                }
                ), 100, !1, this),
                i ? (this.resizeObserver_ = new this.ResizeObserver(this.debouncedHandler_),
                this.resizeObserver_.observe(e.el())) : (this.loadListener_ = ()=>{
                    if (!this.el_ || !this.el_.contentWindow)
                        return;
                    const e = this.debouncedHandler_;
                    let t = this.unloadListener_ = function() {
                        ht(this, "resize", e),
                        ht(this, "unload", t),
                        t = null
                    }
                    ;
                    lt(this.el_.contentWindow, "unload", t),
                    lt(this.el_.contentWindow, "resize", e)
                }
                ,
                this.one("load", this.loadListener_))
            }
            createEl() {
                return super.createEl("iframe", {
                    className: "vjs-resize-manager",
                    tabIndex: -1,
                    title: this.localize("No content")
                }, {
                    "aria-hidden": "true"
                })
            }
            resizeHandler() {
                this.player_ && this.player_.trigger && this.player_.trigger("playerresize")
            }
            dispose() {
                this.debouncedHandler_ && this.debouncedHandler_.cancel(),
                this.resizeObserver_ && (this.player_.el() && this.resizeObserver_.unobserve(this.player_.el()),
                this.resizeObserver_.disconnect()),
                this.loadListener_ && this.off("load", this.loadListener_),
                this.el_ && this.el_.contentWindow && this.unloadListener_ && this.unloadListener_.call(this.el_.contentWindow),
                this.ResizeObserver = null,
                this.resizeObserver = null,
                this.debouncedHandler_ = null,
                this.loadListener_ = null,
                super.dispose()
            }
        }
        );
        const en = {
            trackingThreshold: 20,
            liveTolerance: 15
        };
        Bt.registerComponent("LiveTracker", class extends Bt {
            constructor(e, t) {
                super(e, V(en, t, {
                    createEl: !1
                })),
                this.trackLiveHandler_ = ()=>this.trackLive_(),
                this.handlePlay_ = e=>this.handlePlay(e),
                this.handleFirstTimeupdate_ = e=>this.handleFirstTimeupdate(e),
                this.handleSeeked_ = e=>this.handleSeeked(e),
                this.seekToLiveEdge_ = e=>this.seekToLiveEdge(e),
                this.reset_(),
                this.on(this.player_, "durationchange", (e=>this.handleDurationchange(e))),
                this.on(this.player_, "canplay", (()=>this.toggleTracking()))
            }
            trackLive_() {
                const e = this.player_.seekable();
                if (!e || !e.length)
                    return;
                const t = Number(n().performance.now().toFixed(4))
                  , i = -1 === this.lastTime_ ? 0 : (t - this.lastTime_) / 1e3;
                this.lastTime_ = t,
                this.pastSeekEnd_ = this.pastSeekEnd() + i;
                const s = this.liveCurrentTime()
                  , r = this.player_.currentTime();
                let a = this.player_.paused() || this.seekedBehindLive_ || Math.abs(s - r) > this.options_.liveTolerance;
                this.timeupdateSeen_ && s !== 1 / 0 || (a = !1),
                a !== this.behindLiveEdge_ && (this.behindLiveEdge_ = a,
                this.trigger("liveedgechange"))
            }
            handleDurationchange() {
                this.toggleTracking()
            }
            toggleTracking() {
                this.player_.duration() === 1 / 0 && this.liveWindow() >= this.options_.trackingThreshold ? (this.player_.options_.liveui && this.player_.addClass("vjs-liveui"),
                this.startTracking()) : (this.player_.removeClass("vjs-liveui"),
                this.stopTracking())
            }
            startTracking() {
                this.isTracking() || (this.timeupdateSeen_ || (this.timeupdateSeen_ = this.player_.hasStarted()),
                this.trackingInterval_ = this.setInterval(this.trackLiveHandler_, mt),
                this.trackLive_(),
                this.on(this.player_, ["play", "pause"], this.trackLiveHandler_),
                this.timeupdateSeen_ ? this.on(this.player_, "seeked", this.handleSeeked_) : (this.one(this.player_, "play", this.handlePlay_),
                this.one(this.player_, "timeupdate", this.handleFirstTimeupdate_)))
            }
            handleFirstTimeupdate() {
                this.timeupdateSeen_ = !0,
                this.on(this.player_, "seeked", this.handleSeeked_)
            }
            handleSeeked() {
                const e = Math.abs(this.liveCurrentTime() - this.player_.currentTime());
                this.seekedBehindLive_ = this.nextSeekedFromUser_ && e > 2,
                this.nextSeekedFromUser_ = !1,
                this.trackLive_()
            }
            handlePlay() {
                this.one(this.player_, "timeupdate", this.seekToLiveEdge_)
            }
            reset_() {
                this.lastTime_ = -1,
                this.pastSeekEnd_ = 0,
                this.lastSeekEnd_ = -1,
                this.behindLiveEdge_ = !0,
                this.timeupdateSeen_ = !1,
                this.seekedBehindLive_ = !1,
                this.nextSeekedFromUser_ = !1,
                this.clearInterval(this.trackingInterval_),
                this.trackingInterval_ = null,
                this.off(this.player_, ["play", "pause"], this.trackLiveHandler_),
                this.off(this.player_, "seeked", this.handleSeeked_),
                this.off(this.player_, "play", this.handlePlay_),
                this.off(this.player_, "timeupdate", this.handleFirstTimeupdate_),
                this.off(this.player_, "timeupdate", this.seekToLiveEdge_)
            }
            nextSeekedFromUser() {
                this.nextSeekedFromUser_ = !0
            }
            stopTracking() {
                this.isTracking() && (this.reset_(),
                this.trigger("liveedgechange"))
            }
            seekableEnd() {
                const e = this.player_.seekable()
                  , t = [];
                let i = e ? e.length : 0;
                for (; i--; )
                    t.push(e.end(i));
                return t.length ? t.sort()[t.length - 1] : 1 / 0
            }
            seekableStart() {
                const e = this.player_.seekable()
                  , t = [];
                let i = e ? e.length : 0;
                for (; i--; )
                    t.push(e.start(i));
                return t.length ? t.sort()[0] : 0
            }
            liveWindow() {
                const e = this.liveCurrentTime();
                return e === 1 / 0 ? 0 : e - this.seekableStart()
            }
            isLive() {
                return this.isTracking()
            }
            atLiveEdge() {
                return !this.behindLiveEdge()
            }
            liveCurrentTime() {
                return this.pastSeekEnd() + this.seekableEnd()
            }
            pastSeekEnd() {
                const e = this.seekableEnd();
                return -1 !== this.lastSeekEnd_ && e !== this.lastSeekEnd_ && (this.pastSeekEnd_ = 0),
                this.lastSeekEnd_ = e,
                this.pastSeekEnd_
            }
            behindLiveEdge() {
                return this.behindLiveEdge_
            }
            isTracking() {
                return "number" === typeof this.trackingInterval_
            }
            seekToLiveEdge() {
                this.seekedBehindLive_ = !1,
                this.atLiveEdge() || (this.nextSeekedFromUser_ = !1,
                this.player_.currentTime(this.liveCurrentTime()))
            }
            dispose() {
                this.stopTracking(),
                super.dispose()
            }
        }
        );
        Bt.registerComponent("TitleBar", class extends Bt {
            constructor(e, t) {
                super(e, t),
                this.on("statechanged", (e=>this.updateDom_())),
                this.updateDom_()
            }
            createEl() {
                return this.els = {
                    title: ve("div", {
                        className: "vjs-title-bar-title",
                        id: `vjs-title-bar-title-${st()}`
                    }),
                    description: ve("div", {
                        className: "vjs-title-bar-description",
                        id: `vjs-title-bar-description-${st()}`
                    })
                },
                ve("div", {
                    className: "vjs-title-bar"
                }, {}, z(this.els))
            }
            updateDom_() {
                const e = this.player_.tech_
                  , t = e && e.el_
                  , i = {
                    title: "aria-labelledby",
                    description: "aria-describedby"
                };
                ["title", "description"].forEach((e=>{
                    const s = this.state[e]
                      , n = this.els[e]
                      , r = i[e];
                    Be(n),
                    s && Te(n, s),
                    t && (t.removeAttribute(r),
                    s && t.setAttribute(r, n.id))
                }
                )),
                this.state.title || this.state.description ? this.show() : this.hide()
            }
            update(e) {
                this.setState(e)
            }
            dispose() {
                const e = this.player_.tech_
                  , t = e && e.el_;
                t && (t.removeAttribute("aria-labelledby"),
                t.removeAttribute("aria-describedby")),
                super.dispose(),
                this.els = null
            }
        }
        );
        const tn = e=>{
            const t = e.el();
            if (t.hasAttribute("src"))
                return e.triggerSourceset(t.src),
                !0;
            const i = e.$$("source")
              , s = [];
            let n = "";
            if (!i.length)
                return !1;
            for (let r = 0; r < i.length; r++) {
                const e = i[r].src;
                e && -1 === s.indexOf(e) && s.push(e)
            }
            return !!s.length && (1 === s.length && (n = s[0]),
            e.triggerSourceset(n),
            !0)
        }
          , sn = Object.defineProperty({}, "innerHTML", {
            get() {
                return this.cloneNode(!0).innerHTML
            },
            set(e) {
                const t = a().createElement(this.nodeName.toLowerCase());
                t.innerHTML = e;
                const i = a().createDocumentFragment();
                for (; t.childNodes.length; )
                    i.appendChild(t.childNodes[0]);
                return this.innerText = "",
                n().Element.prototype.appendChild.call(this, i),
                this.innerHTML
            }
        })
          , nn = (e,t)=>{
            let i = {};
            for (let s = 0; s < e.length && (i = Object.getOwnPropertyDescriptor(e[s], t),
            !(i && i.set && i.get)); s++)
                ;
            return i.enumerable = !0,
            i.configurable = !0,
            i
        }
          , rn = function(e) {
            const t = e.el();
            if (t.resetSourceWatch_)
                return;
            const i = {}
              , s = (e=>nn([e.el(), n().HTMLMediaElement.prototype, n().Element.prototype, sn], "innerHTML"))(e)
              , r = i=>(...s)=>{
                const n = i.apply(t, s);
                return tn(e),
                n
            }
            ;
            ["append", "appendChild", "insertAdjacentHTML"].forEach((e=>{
                t[e] && (i[e] = t[e],
                t[e] = r(i[e]))
            }
            )),
            Object.defineProperty(t, "innerHTML", V(s, {
                set: r(s.set)
            })),
            t.resetSourceWatch_ = ()=>{
                t.resetSourceWatch_ = null,
                Object.keys(i).forEach((e=>{
                    t[e] = i[e]
                }
                )),
                Object.defineProperty(t, "innerHTML", s)
            }
            ,
            e.one("sourceset", t.resetSourceWatch_)
        }
          , an = Object.defineProperty({}, "src", {
            get() {
                return this.hasAttribute("src") ? ci(n().Element.prototype.getAttribute.call(this, "src")) : ""
            },
            set(e) {
                return n().Element.prototype.setAttribute.call(this, "src", e),
                e
            }
        })
          , on = function(e) {
            if (!e.featuresSourceset)
                return;
            const t = e.el();
            if (t.resetSourceset_)
                return;
            const i = (e=>nn([e.el(), n().HTMLMediaElement.prototype, an], "src"))(e)
              , s = t.setAttribute
              , r = t.load;
            Object.defineProperty(t, "src", V(i, {
                set: s=>{
                    const n = i.set.call(t, s);
                    return e.triggerSourceset(t.src),
                    n
                }
            })),
            t.setAttribute = (i,n)=>{
                const r = s.call(t, i, n);
                return /src/i.test(i) && e.triggerSourceset(t.src),
                r
            }
            ,
            t.load = ()=>{
                const i = r.call(t);
                return tn(e) || (e.triggerSourceset(""),
                rn(e)),
                i
            }
            ,
            t.currentSrc ? e.triggerSourceset(t.currentSrc) : tn(e) || rn(e),
            t.resetSourceset_ = ()=>{
                t.resetSourceset_ = null,
                t.load = r,
                t.setAttribute = s,
                Object.defineProperty(t, "src", i),
                t.resetSourceWatch_ && t.resetSourceWatch_()
            }
        };
        class ln extends Ei {
            constructor(e, t) {
                super(e, t);
                const i = e.source;
                let s = !1;
                if (this.featuresVideoFrameCallback = this.featuresVideoFrameCallback && "VIDEO" === this.el_.tagName,
                i && (this.el_.currentSrc !== i.src || e.tag && 3 === e.tag.initNetworkState_) ? this.setSource(i) : this.handleLateInit_(this.el_),
                e.enableSourceset && this.setupSourcesetHandling_(),
                this.isScrubbing_ = !1,
                this.el_.hasChildNodes()) {
                    const e = this.el_.childNodes;
                    let t = e.length;
                    const i = [];
                    for (; t--; ) {
                        const n = e[t];
                        "track" === n.nodeName.toLowerCase() && (this.featuresNativeTextTracks ? (this.remoteTextTrackEls().addTrackElement_(n),
                        this.remoteTextTracks().addTrack(n.track),
                        this.textTracks().addTrack(n.track),
                        s || this.el_.hasAttribute("crossorigin") || !mi(n.src) || (s = !0)) : i.push(n))
                    }
                    for (let s = 0; s < i.length; s++)
                        this.el_.removeChild(i[s])
                }
                this.proxyNativeTracks_(),
                this.featuresNativeTextTracks && s && U.warn("Text Tracks are being loaded from another origin but the crossorigin attribute isn't used.\nThis may prevent text tracks from loading."),
                this.restoreMetadataTracksInIOSNativePlayer_(),
                (he || le) && !0 === e.nativeControlsForTouch && this.setControls(!0),
                this.proxyWebkitFullscreen_(),
                this.triggerReady()
            }
            dispose() {
                this.el_ && this.el_.resetSourceset_ && this.el_.resetSourceset_(),
                ln.disposeMediaElement(this.el_),
                this.options_ = null,
                super.dispose()
            }
            setupSourcesetHandling_() {
                on(this)
            }
            restoreMetadataTracksInIOSNativePlayer_() {
                const e = this.textTracks();
                let t;
                const i = ()=>{
                    t = [];
                    for (let i = 0; i < e.length; i++) {
                        const s = e[i];
                        "metadata" === s.kind && t.push({
                            track: s,
                            storedMode: s.mode
                        })
                    }
                }
                ;
                i(),
                e.addEventListener("change", i),
                this.on("dispose", (()=>e.removeEventListener("change", i)));
                const s = ()=>{
                    for (let e = 0; e < t.length; e++) {
                        const i = t[e];
                        "disabled" === i.track.mode && i.track.mode !== i.storedMode && (i.track.mode = i.storedMode)
                    }
                    e.removeEventListener("change", s)
                }
                ;
                this.on("webkitbeginfullscreen", (()=>{
                    e.removeEventListener("change", i),
                    e.removeEventListener("change", s),
                    e.addEventListener("change", s)
                }
                )),
                this.on("webkitendfullscreen", (()=>{
                    e.removeEventListener("change", i),
                    e.addEventListener("change", i),
                    e.removeEventListener("change", s)
                }
                ))
            }
            overrideNative_(e, t) {
                if (t !== this[`featuresNative${e}Tracks`])
                    return;
                const i = e.toLowerCase();
                this[`${i}TracksListeners_`] && Object.keys(this[`${i}TracksListeners_`]).forEach((e=>{
                    this.el()[`${i}Tracks`].removeEventListener(e, this[`${i}TracksListeners_`][e])
                }
                )),
                this[`featuresNative${e}Tracks`] = !t,
                this[`${i}TracksListeners_`] = null,
                this.proxyNativeTracksForType_(i)
            }
            overrideNativeAudioTracks(e) {
                this.overrideNative_("Audio", e)
            }
            overrideNativeVideoTracks(e) {
                this.overrideNative_("Video", e)
            }
            proxyNativeTracksForType_(e) {
                const t = Si[e]
                  , i = this.el()[t.getterName]
                  , s = this[t.getterName]();
                if (!this[`featuresNative${t.capitalName}Tracks`] || !i || !i.addEventListener)
                    return;
                const n = {
                    change: t=>{
                        const i = {
                            type: "change",
                            target: s,
                            currentTarget: s,
                            srcElement: s
                        };
                        s.trigger(i),
                        "text" === e && this[ki.remoteText.getterName]().trigger(i)
                    }
                    ,
                    addtrack(e) {
                        s.addTrack(e.track)
                    },
                    removetrack(e) {
                        s.removeTrack(e.track)
                    }
                }
                  , r = function() {
                    const e = [];
                    for (let t = 0; t < s.length; t++) {
                        let n = !1;
                        for (let e = 0; e < i.length; e++)
                            if (i[e] === s[t]) {
                                n = !0;
                                break
                            }
                        n || e.push(s[t])
                    }
                    for (; e.length; )
                        s.removeTrack(e.shift())
                };
                this[t.getterName + "Listeners_"] = n,
                Object.keys(n).forEach((e=>{
                    const t = n[e];
                    i.addEventListener(e, t),
                    this.on("dispose", (s=>i.removeEventListener(e, t)))
                }
                )),
                this.on("loadstart", r),
                this.on("dispose", (e=>this.off("loadstart", r)))
            }
            proxyNativeTracks_() {
                Si.names.forEach((e=>{
                    this.proxyNativeTracksForType_(e)
                }
                ))
            }
            createEl() {
                let e = this.options_.tag;
                if (!e || !this.options_.playerElIngest && !this.movingMediaElementInDOM) {
                    if (e) {
                        const t = e.cloneNode(!0);
                        e.parentNode && e.parentNode.insertBefore(t, e),
                        ln.disposeMediaElement(e),
                        e = t
                    } else {
                        e = a().createElement("video");
                        const t = V({}, this.options_.tag && xe(this.options_.tag));
                        he && !0 === this.options_.nativeControlsForTouch || delete t.controls,
                        we(e, Object.assign(t, {
                            id: this.options_.techId,
                            class: "vjs-tech"
                        }))
                    }
                    e.playerId = this.options_.playerId
                }
                "undefined" !== typeof this.options_.preload && Pe(e, "preload", this.options_.preload),
                void 0 !== this.options_.disablePictureInPicture && (e.disablePictureInPicture = this.options_.disablePictureInPicture);
                const t = ["loop", "muted", "playsinline", "autoplay"];
                for (let i = 0; i < t.length; i++) {
                    const s = t[i]
                      , n = this.options_[s];
                    "undefined" !== typeof n && (n ? Pe(e, s, s) : Ae(e, s),
                    e[s] = n)
                }
                return e
            }
            handleLateInit_(e) {
                if (0 === e.networkState || 3 === e.networkState)
                    return;
                if (0 === e.readyState) {
                    let e = !1;
                    const t = function() {
                        e = !0
                    };
                    this.on("loadstart", t);
                    const i = function() {
                        e || this.trigger("loadstart")
                    };
                    return this.on("loadedmetadata", i),
                    void this.ready((function() {
                        this.off("loadstart", t),
                        this.off("loadedmetadata", i),
                        e || this.trigger("loadstart")
                    }
                    ))
                }
                const t = ["loadstart"];
                t.push("loadedmetadata"),
                e.readyState >= 2 && t.push("loadeddata"),
                e.readyState >= 3 && t.push("canplay"),
                e.readyState >= 4 && t.push("canplaythrough"),
                this.ready((function() {
                    t.forEach((function(e) {
                        this.trigger(e)
                    }
                    ), this)
                }
                ))
            }
            setScrubbing(e) {
                this.isScrubbing_ = e
            }
            scrubbing() {
                return this.isScrubbing_
            }
            setCurrentTime(e) {
                try {
                    this.isScrubbing_ && this.el_.fastSeek && ce ? this.el_.fastSeek(e) : this.el_.currentTime = e
                } catch (t) {
                    U(t, "Video is not ready. (Video.js)")
                }
            }
            duration() {
                if (this.el_.duration === 1 / 0 && Y && te && 0 === this.el_.currentTime) {
                    const e = ()=>{
                        this.el_.currentTime > 0 && (this.el_.duration === 1 / 0 && this.trigger("durationchange"),
                        this.off("timeupdate", e))
                    }
                    ;
                    return this.on("timeupdate", e),
                    NaN
                }
                return this.el_.duration || NaN
            }
            width() {
                return this.el_.offsetWidth
            }
            height() {
                return this.el_.offsetHeight
            }
            proxyWebkitFullscreen_() {
                if (!("webkitDisplayingFullscreen"in this.el_))
                    return;
                const e = function() {
                    this.trigger("fullscreenchange", {
                        isFullscreen: !1
                    }),
                    this.el_.controls && !this.options_.nativeControlsForTouch && this.controls() && (this.el_.controls = !1)
                }
                  , t = function() {
                    "webkitPresentationMode"in this.el_ && "picture-in-picture" !== this.el_.webkitPresentationMode && (this.one("webkitendfullscreen", e),
                    this.trigger("fullscreenchange", {
                        isFullscreen: !0,
                        nativeIOSFullscreen: !0
                    }))
                };
                this.on("webkitbeginfullscreen", t),
                this.on("dispose", (()=>{
                    this.off("webkitbeginfullscreen", t),
                    this.off("webkitendfullscreen", e)
                }
                ))
            }
            supportsFullScreen() {
                return "function" === typeof this.el_.webkitEnterFullScreen
            }
            enterFullScreen() {
                const e = this.el_;
                if (e.paused && e.networkState <= e.HAVE_METADATA)
                    Xt(this.el_.play()),
                    this.setTimeout((function() {
                        e.pause();
                        try {
                            e.webkitEnterFullScreen()
                        } catch (t) {
                            this.trigger("fullscreenerror", t)
                        }
                    }
                    ), 0);
                else
                    try {
                        e.webkitEnterFullScreen()
                    } catch (t) {
                        this.trigger("fullscreenerror", t)
                    }
            }
            exitFullScreen() {
                this.el_.webkitDisplayingFullscreen ? this.el_.webkitExitFullScreen() : this.trigger("fullscreenerror", new Error("The video is not fullscreen"))
            }
            requestPictureInPicture() {
                return this.el_.requestPictureInPicture()
            }
            requestVideoFrameCallback(e) {
                return this.featuresVideoFrameCallback && !this.el_.webkitKeys ? this.el_.requestVideoFrameCallback(e) : super.requestVideoFrameCallback(e)
            }
            cancelVideoFrameCallback(e) {
                this.featuresVideoFrameCallback && !this.el_.webkitKeys ? this.el_.cancelVideoFrameCallback(e) : super.cancelVideoFrameCallback(e)
            }
            src(e) {
                if (void 0 === e)
                    return this.el_.src;
                this.setSrc(e)
            }
            reset() {
                ln.resetMediaElement(this.el_)
            }
            currentSrc() {
                return this.currentSource_ ? this.currentSource_.src : this.el_.currentSrc
            }
            setControls(e) {
                this.el_.controls = !!e
            }
            addTextTrack(e, t, i) {
                return this.featuresNativeTextTracks ? this.el_.addTextTrack(e, t, i) : super.addTextTrack(e, t, i)
            }
            createRemoteTextTrack(e) {
                if (!this.featuresNativeTextTracks)
                    return super.createRemoteTextTrack(e);
                const t = a().createElement("track");
                return e.kind && (t.kind = e.kind),
                e.label && (t.label = e.label),
                (e.language || e.srclang) && (t.srclang = e.language || e.srclang),
                e.default && (t.default = e.default),
                e.id && (t.id = e.id),
                e.src && (t.src = e.src),
                t
            }
            addRemoteTextTrack(e, t) {
                const i = super.addRemoteTextTrack(e, t);
                return this.featuresNativeTextTracks && this.el().appendChild(i),
                i
            }
            removeRemoteTextTrack(e) {
                if (super.removeRemoteTextTrack(e),
                this.featuresNativeTextTracks) {
                    const t = this.$$("track");
                    let i = t.length;
                    for (; i--; )
                        e !== t[i] && e !== t[i].track || this.el().removeChild(t[i])
                }
            }
            getVideoPlaybackQuality() {
                if ("function" === typeof this.el().getVideoPlaybackQuality)
                    return this.el().getVideoPlaybackQuality();
                const e = {};
                return "undefined" !== typeof this.el().webkitDroppedFrameCount && "undefined" !== typeof this.el().webkitDecodedFrameCount && (e.droppedVideoFrames = this.el().webkitDroppedFrameCount,
                e.totalVideoFrames = this.el().webkitDecodedFrameCount),
                n().performance && (e.creationTime = n().performance.now()),
                e
            }
        }
        W(ln, "TEST_VID", (function() {
            if (!ge())
                return;
            const e = a().createElement("video")
              , t = a().createElement("track");
            return t.kind = "captions",
            t.srclang = "en",
            t.label = "English",
            e.appendChild(t),
            e
        }
        )),
        ln.isSupported = function() {
            try {
                ln.TEST_VID.volume = .5
            } catch (e) {
                return !1
            }
            return !(!ln.TEST_VID || !ln.TEST_VID.canPlayType)
        }
        ,
        ln.canPlayType = function(e) {
            return ln.TEST_VID.canPlayType(e)
        }
        ,
        ln.canPlaySource = function(e, t) {
            return ln.canPlayType(e.type)
        }
        ,
        ln.canControlVolume = function() {
            try {
                const e = ln.TEST_VID.volume;
                ln.TEST_VID.volume = e / 2 + .1;
                const t = e !== ln.TEST_VID.volume;
                return t && ue ? (n().setTimeout((()=>{
                    ln && ln.prototype && (ln.prototype.featuresVolumeControl = e !== ln.TEST_VID.volume)
                }
                )),
                !1) : t
            } catch (e) {
                return !1
            }
        }
        ,
        ln.canMuteVolume = function() {
            try {
                const e = ln.TEST_VID.muted;
                return ln.TEST_VID.muted = !e,
                ln.TEST_VID.muted ? Pe(ln.TEST_VID, "muted", "muted") : Ae(ln.TEST_VID, "muted"),
                e !== ln.TEST_VID.muted
            } catch (e) {
                return !1
            }
        }
        ,
        ln.canControlPlaybackRate = function() {
            if (Y && te && se < 58)
                return !1;
            try {
                const e = ln.TEST_VID.playbackRate;
                return ln.TEST_VID.playbackRate = e / 2 + .1,
                e !== ln.TEST_VID.playbackRate
            } catch (e) {
                return !1
            }
        }
        ,
        ln.canOverrideAttributes = function() {
            try {
                const e = ()=>{}
                ;
                Object.defineProperty(a().createElement("video"), "src", {
                    get: e,
                    set: e
                }),
                Object.defineProperty(a().createElement("audio"), "src", {
                    get: e,
                    set: e
                }),
                Object.defineProperty(a().createElement("video"), "innerHTML", {
                    get: e,
                    set: e
                }),
                Object.defineProperty(a().createElement("audio"), "innerHTML", {
                    get: e,
                    set: e
                })
            } catch (e) {
                return !1
            }
            return !0
        }
        ,
        ln.supportsNativeTextTracks = function() {
            return ce || ue && te
        }
        ,
        ln.supportsNativeVideoTracks = function() {
            return !(!ln.TEST_VID || !ln.TEST_VID.videoTracks)
        }
        ,
        ln.supportsNativeAudioTracks = function() {
            return !(!ln.TEST_VID || !ln.TEST_VID.audioTracks)
        }
        ,
        ln.Events = ["loadstart", "suspend", "abort", "error", "emptied", "stalled", "loadedmetadata", "loadeddata", "canplay", "canplaythrough", "playing", "waiting", "seeking", "seeked", "ended", "durationchange", "timeupdate", "progress", "play", "pause", "ratechange", "resize", "volumechange"],
        [["featuresMuteControl", "canMuteVolume"], ["featuresPlaybackRate", "canControlPlaybackRate"], ["featuresSourceset", "canOverrideAttributes"], ["featuresNativeTextTracks", "supportsNativeTextTracks"], ["featuresNativeVideoTracks", "supportsNativeVideoTracks"], ["featuresNativeAudioTracks", "supportsNativeAudioTracks"]].forEach((function([e,t]) {
            W(ln.prototype, e, (()=>ln[t]()), !0)
        }
        )),
        ln.prototype.featuresVolumeControl = ln.canControlVolume(),
        ln.prototype.movingMediaElementInDOM = !ue,
        ln.prototype.featuresFullscreenResize = !0,
        ln.prototype.featuresProgressEvents = !0,
        ln.prototype.featuresTimeupdateEvents = !0,
        ln.prototype.featuresVideoFrameCallback = !(!ln.TEST_VID || !ln.TEST_VID.requestVideoFrameCallback),
        ln.disposeMediaElement = function(e) {
            if (e) {
                for (e.parentNode && e.parentNode.removeChild(e); e.hasChildNodes(); )
                    e.removeChild(e.firstChild);
                e.removeAttribute("src"),
                "function" === typeof e.load && function() {
                    try {
                        e.load()
                    } catch (t) {}
                }()
            }
        }
        ,
        ln.resetMediaElement = function(e) {
            if (!e)
                return;
            const t = e.querySelectorAll("source");
            let i = t.length;
            for (; i--; )
                e.removeChild(t[i]);
            e.removeAttribute("src"),
            "function" === typeof e.load && function() {
                try {
                    e.load()
                } catch (t) {}
            }()
        }
        ,
        ["muted", "defaultMuted", "autoplay", "controls", "loop", "playsinline"].forEach((function(e) {
            ln.prototype[e] = function() {
                return this.el_[e] || this.el_.hasAttribute(e)
            }
        }
        )),
        ["muted", "defaultMuted", "autoplay", "loop", "playsinline"].forEach((function(e) {
            ln.prototype["set" + Mt(e)] = function(t) {
                this.el_[e] = t,
                t ? this.el_.setAttribute(e, e) : this.el_.removeAttribute(e)
            }
        }
        )),
        ["paused", "currentTime", "buffered", "volume", "poster", "preload", "error", "seeking", "seekable", "ended", "playbackRate", "defaultPlaybackRate", "disablePictureInPicture", "played", "networkState", "readyState", "videoWidth", "videoHeight", "crossOrigin"].forEach((function(e) {
            ln.prototype[e] = function() {
                return this.el_[e]
            }
        }
        )),
        ["volume", "src", "poster", "preload", "playbackRate", "defaultPlaybackRate", "disablePictureInPicture", "crossOrigin"].forEach((function(e) {
            ln.prototype["set" + Mt(e)] = function(t) {
                this.el_[e] = t
            }
        }
        )),
        ["pause", "load", "play"].forEach((function(e) {
            ln.prototype[e] = function() {
                return this.el_[e]()
            }
        }
        )),
        Ei.withSourceHandlers(ln),
        ln.nativeSourceHandler = {},
        ln.nativeSourceHandler.canPlayType = function(e) {
            try {
                return ln.TEST_VID.canPlayType(e)
            } catch (t) {
                return ""
            }
        }
        ,
        ln.nativeSourceHandler.canHandleSource = function(e, t) {
            if (e.type)
                return ln.nativeSourceHandler.canPlayType(e.type);
            if (e.src) {
                const t = pi(e.src);
                return ln.nativeSourceHandler.canPlayType(`video/${t}`)
            }
            return ""
        }
        ,
        ln.nativeSourceHandler.handleSource = function(e, t, i) {
            t.setSrc(e.src)
        }
        ,
        ln.nativeSourceHandler.dispose = function() {}
        ,
        ln.registerSourceHandler(ln.nativeSourceHandler),
        Ei.registerTech("Html5", ln);
        const hn = ["progress", "abort", "suspend", "emptied", "stalled", "loadedmetadata", "loadeddata", "timeupdate", "resize", "volumechange", "texttrackchange"]
          , dn = {
            canplay: "CanPlay",
            canplaythrough: "CanPlayThrough",
            playing: "Playing",
            seeked: "Seeked"
        }
          , un = ["tiny", "xsmall", "small", "medium", "large", "xlarge", "huge"]
          , cn = {};
        un.forEach((e=>{
            const t = "x" === e.charAt(0) ? `x-${e.substring(1)}` : e;
            cn[e] = `vjs-layout-${t}`
        }
        ));
        const pn = {
            tiny: 210,
            xsmall: 320,
            small: 425,
            medium: 768,
            large: 1440,
            xlarge: 2560,
            huge: 1 / 0
        };
        class mn extends Bt {
            constructor(e, t, i) {
                if (e.id = e.id || t.id || `vjs_video_${st()}`,
                (t = Object.assign(mn.getTagSettings(e), t)).initChildren = !1,
                t.createEl = !1,
                t.evented = !1,
                t.reportTouchActivity = !1,
                !t.language) {
                    const i = e.closest("[lang]");
                    i && (t.language = i.getAttribute("lang"))
                }
                if (super(null, t, i),
                this.boundDocumentFullscreenChange_ = e=>this.documentFullscreenChange_(e),
                this.boundFullWindowOnEscKey_ = e=>this.fullWindowOnEscKey(e),
                this.boundUpdateStyleEl_ = e=>this.updateStyleEl_(e),
                this.boundApplyInitTime_ = e=>this.applyInitTime_(e),
                this.boundUpdateCurrentBreakpoint_ = e=>this.updateCurrentBreakpoint_(e),
                this.boundHandleTechClick_ = e=>this.handleTechClick_(e),
                this.boundHandleTechDoubleClick_ = e=>this.handleTechDoubleClick_(e),
                this.boundHandleTechTouchStart_ = e=>this.handleTechTouchStart_(e),
                this.boundHandleTechTouchMove_ = e=>this.handleTechTouchMove_(e),
                this.boundHandleTechTouchEnd_ = e=>this.handleTechTouchEnd_(e),
                this.boundHandleTechTap_ = e=>this.handleTechTap_(e),
                this.isFullscreen_ = !1,
                this.log = B(this.id_),
                this.fsApi_ = L,
                this.isPosterFromTech_ = !1,
                this.queuedCallbacks_ = [],
                this.isReady_ = !1,
                this.hasStarted_ = !1,
                this.userActive_ = !1,
                this.debugEnabled_ = !1,
                this.audioOnlyMode_ = !1,
                this.audioPosterMode_ = !1,
                this.audioOnlyCache_ = {
                    playerHeight: null,
                    hiddenChildren: []
                },
                !this.options_ || !this.options_.techOrder || !this.options_.techOrder.length)
                    throw new Error("No techOrder specified. Did you overwrite videojs.options instead of just changing the properties you want to override?");
                if (this.tag = e,
                this.tagAttributes = e && xe(e),
                this.language(this.options_.language),
                t.languages) {
                    const e = {};
                    Object.getOwnPropertyNames(t.languages).forEach((function(i) {
                        e[i.toLowerCase()] = t.languages[i]
                    }
                    )),
                    this.languages_ = e
                } else
                    this.languages_ = mn.prototype.options_.languages;
                this.resetCache_(),
                this.poster_ = t.poster || "",
                this.controls_ = !!t.controls,
                e.controls = !1,
                e.removeAttribute("controls"),
                this.changingSrc_ = !1,
                this.playCallbacks_ = [],
                this.playTerminatedQueue_ = [],
                e.hasAttribute("autoplay") ? this.autoplay(!0) : this.autoplay(this.options_.autoplay),
                t.plugins && Object.keys(t.plugins).forEach((e=>{
                    if ("function" !== typeof this[e])
                        throw new Error(`plugin "${e}" does not exist`)
                }
                )),
                this.scrubbing_ = !1,
                this.el_ = this.createEl(),
                At(this, {
                    eventBusKey: "el_"
                }),
                this.fsApi_.requestFullscreen && (lt(a(), this.fsApi_.fullscreenchange, this.boundDocumentFullscreenChange_),
                this.on(this.fsApi_.fullscreenchange, this.boundDocumentFullscreenChange_)),
                this.fluid_ && this.on(["playerreset", "resize"], this.boundUpdateStyleEl_);
                const s = V(this.options_);
                if (t.plugins && Object.keys(t.plugins).forEach((e=>{
                    this[e](t.plugins[e])
                }
                )),
                t.debug && this.debug(!0),
                this.options_.playerOptions = s,
                this.middleware_ = [],
                this.playbackRates(t.playbackRates),
                t.experimentalSvgIcons) {
                    const e = (new (n().DOMParser)).parseFromString('<svg xmlns="http://www.w3.org/2000/svg">\n  <defs>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-play">\n      <path d="M16 10v28l22-14z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-pause">\n      <path d="M12 38h8V10h-8v28zm16-28v28h8V10h-8z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-audio">\n      <path d="M24 2C14.06 2 6 10.06 6 20v14c0 3.31 2.69 6 6 6h6V24h-8v-4c0-7.73 6.27-14 14-14s14 6.27 14 14v4h-8v16h6c3.31 0 6-2.69 6-6V20c0-9.94-8.06-18-18-18z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-captions">\n      <path d="M38 8H10c-2.21 0-4 1.79-4 4v24c0 2.21 1.79 4 4 4h28c2.21 0 4-1.79 4-4V12c0-2.21-1.79-4-4-4zM22 22h-3v-1h-4v6h4v-1h3v2a2 2 0 0 1-2 2h-6a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2zm14 0h-3v-1h-4v6h4v-1h3v2a2 2 0 0 1-2 2h-6a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-subtitles">\n      <path d="M40 8H8c-2.21 0-4 1.79-4 4v24c0 2.21 1.79 4 4 4h32c2.21 0 4-1.79 4-4V12c0-2.21-1.79-4-4-4zM8 24h8v4H8v-4zm20 12H8v-4h20v4zm12 0h-8v-4h8v4zm0-8H20v-4h20v4z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-fullscreen-enter">\n      <path d="M14 28h-4v10h10v-4h-6v-6zm-4-8h4v-6h6v-4H10v10zm24 14h-6v4h10V28h-4v6zm-6-24v4h6v6h4V10H28z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-fullscreen-exit">\n      <path d="M10 32h6v6h4V28H10v4zm6-16h-6v4h10V10h-4v6zm12 22h4v-6h6v-4H28v10zm4-22v-6h-4v10h10v-4h-6z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-play-circle">\n      <path d="M20 33l12-9-12-9v18zm4-29C12.95 4 4 12.95 4 24s8.95 20 20 20 20-8.95 20-20S35.05 4 24 4zm0 36c-8.82 0-16-7.18-16-16S15.18 8 24 8s16 7.18 16 16-7.18 16-16 16z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-volume-mute">\n      <path d="M33 24c0-3.53-2.04-6.58-5-8.05v4.42l4.91 4.91c.06-.42.09-.85.09-1.28zm5 0c0 1.88-.41 3.65-1.08 5.28l3.03 3.03C41.25 29.82 42 27 42 24c0-8.56-5.99-15.72-14-17.54v4.13c5.78 1.72 10 7.07 10 13.41zM8.55 6L6 8.55 15.45 18H6v12h8l10 10V26.55l8.51 8.51c-1.34 1.03-2.85 1.86-4.51 2.36v4.13a17.94 17.94 0 0 0 7.37-3.62L39.45 42 42 39.45l-18-18L8.55 6zM24 8l-4.18 4.18L24 16.36V8z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-volume-low">\n      <path d="M14 18v12h8l10 10V8L22 18h-8z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-volume-medium">\n      <path d="M37 24c0-3.53-2.04-6.58-5-8.05v16.11c2.96-1.48 5-4.53 5-8.06zm-27-6v12h8l10 10V8L18 18h-8z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-volume-high">\n      <path d="M6 18v12h8l10 10V8L14 18H6zm27 6c0-3.53-2.04-6.58-5-8.05v16.11c2.96-1.48 5-4.53 5-8.06zM28 6.46v4.13c5.78 1.72 10 7.07 10 13.41s-4.22 11.69-10 13.41v4.13c8.01-1.82 14-8.97 14-17.54S36.01 8.28 28 6.46z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-spinner">\n      <path d="M18.8 21l9.53-16.51C26.94 4.18 25.49 4 24 4c-4.8 0-9.19 1.69-12.64 4.51l7.33 12.69.11-.2zm24.28-3c-1.84-5.85-6.3-10.52-11.99-12.68L23.77 18h19.31zm.52 2H28.62l.58 1 9.53 16.5C41.99 33.94 44 29.21 44 24c0-1.37-.14-2.71-.4-4zm-26.53 4l-7.8-13.5C6.01 14.06 4 18.79 4 24c0 1.37.14 2.71.4 4h14.98l-2.31-4zM4.92 30c1.84 5.85 6.3 10.52 11.99 12.68L24.23 30H4.92zm22.54 0l-7.8 13.51c1.4.31 2.85.49 4.34.49 4.8 0 9.19-1.69 12.64-4.51L29.31 26.8 27.46 30z"></path>\n    </symbol>\n    <symbol viewBox="0 0 24 24" id="vjs-icon-hd">\n      <path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-8 12H9.5v-2h-2v2H6V9h1.5v2.5h2V9H11v6zm2-6h4c.55 0 1 .45 1 1v4c0 .55-.45 1-1 1h-4V9zm1.5 4.5h2v-3h-2v3z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-chapters">\n      <path d="M6 26h4v-4H6v4zm0 8h4v-4H6v4zm0-16h4v-4H6v4zm8 8h28v-4H14v4zm0 8h28v-4H14v4zm0-20v4h28v-4H14z"></path>\n    </symbol>\n    <symbol viewBox="0 0 40 40" id="vjs-icon-downloading">\n      <path d="M18.208 36.875q-3.208-.292-5.979-1.729-2.771-1.438-4.812-3.729-2.042-2.292-3.188-5.229-1.146-2.938-1.146-6.23 0-6.583 4.334-11.416 4.333-4.834 10.833-5.5v3.166q-5.167.75-8.583 4.646Q6.25 14.75 6.25 19.958q0 5.209 3.396 9.104 3.396 3.896 8.562 4.646zM20 28.417L11.542 20l2.083-2.083 4.917 4.916v-11.25h2.916v11.25l4.875-4.916L28.417 20zm1.792 8.458v-3.167q1.833-.25 3.541-.958 1.709-.708 3.167-1.875l2.333 2.292q-1.958 1.583-4.25 2.541-2.291.959-4.791 1.167zm6.791-27.792q-1.541-1.125-3.25-1.854-1.708-.729-3.541-1.021V3.042q2.5.25 4.77 1.208 2.271.958 4.271 2.5zm4.584 21.584l-2.25-2.25q1.166-1.5 1.854-3.209.687-1.708.937-3.541h3.209q-.292 2.5-1.229 4.791-.938 2.292-2.521 4.209zm.541-12.417q-.291-1.833-.958-3.562-.667-1.73-1.833-3.188l2.375-2.208q1.541 1.916 2.458 4.208.917 2.292 1.167 4.75z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-file-download">\n      <path d="M10.8 40.55q-1.35 0-2.375-1T7.4 37.15v-7.7h3.4v7.7h26.35v-7.7h3.4v7.7q0 1.4-1 2.4t-2.4 1zM24 32.1L13.9 22.05l2.45-2.45 5.95 5.95V7.15h3.4v18.4l5.95-5.95 2.45 2.45z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-file-download-done">\n      <path d="M9.8 40.5v-3.45h28.4v3.45zm9.2-9.05L7.4 19.85l2.45-2.35L19 26.65l19.2-19.2 2.4 2.4z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-file-download-off">\n      <path d="M4.9 4.75L43.25 43.1 41 45.3l-4.75-4.75q-.05.05-.075.025-.025-.025-.075-.025H10.8q-1.35 0-2.375-1T7.4 37.15v-7.7h3.4v7.7h22.05l-7-7-1.85 1.8L13.9 21.9l1.85-1.85L2.7 7zm26.75 14.7l2.45 2.45-3.75 3.8-2.45-2.5zM25.7 7.15V21.1l-3.4-3.45V7.15z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-share">\n      <path d="M36 32.17c-1.52 0-2.89.59-3.93 1.54L17.82 25.4c.11-.45.18-.92.18-1.4s-.07-.95-.18-1.4l14.1-8.23c1.07 1 2.5 1.62 4.08 1.62 3.31 0 6-2.69 6-6s-2.69-6-6-6-6 2.69-6 6c0 .48.07.95.18 1.4l-14.1 8.23c-1.07-1-2.5-1.62-4.08-1.62-3.31 0-6 2.69-6 6s2.69 6 6 6c1.58 0 3.01-.62 4.08-1.62l14.25 8.31c-.1.42-.16.86-.16 1.31A5.83 5.83 0 1 0 36 32.17z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-cog">\n      <path d="M38.86 25.95c.08-.64.14-1.29.14-1.95s-.06-1.31-.14-1.95l4.23-3.31c.38-.3.49-.84.24-1.28l-4-6.93c-.25-.43-.77-.61-1.22-.43l-4.98 2.01c-1.03-.79-2.16-1.46-3.38-1.97L29 4.84c-.09-.47-.5-.84-1-.84h-8c-.5 0-.91.37-.99.84l-.75 5.3a14.8 14.8 0 0 0-3.38 1.97L9.9 10.1a1 1 0 0 0-1.22.43l-4 6.93c-.25.43-.14.97.24 1.28l4.22 3.31C9.06 22.69 9 23.34 9 24s.06 1.31.14 1.95l-4.22 3.31c-.38.3-.49.84-.24 1.28l4 6.93c.25.43.77.61 1.22.43l4.98-2.01c1.03.79 2.16 1.46 3.38 1.97l.75 5.3c.08.47.49.84.99.84h8c.5 0 .91-.37.99-.84l.75-5.3a14.8 14.8 0 0 0 3.38-1.97l4.98 2.01a1 1 0 0 0 1.22-.43l4-6.93c.25-.43.14-.97-.24-1.28l-4.22-3.31zM24 31c-3.87 0-7-3.13-7-7s3.13-7 7-7 7 3.13 7 7-3.13 7-7 7z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-square">\n      <path d="M36 8H12c-2.21 0-4 1.79-4 4v24c0 2.21 1.79 4 4 4h24c2.21 0 4-1.79 4-4V12c0-2.21-1.79-4-4-4zm0 28H12V12h24v24z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-circle">\n      <circle cx="24" cy="24" r="20"></circle>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-circle-outline">\n      <path d="M24 4C12.95 4 4 12.95 4 24s8.95 20 20 20 20-8.95 20-20S35.05 4 24 4zm0 36c-8.82 0-16-7.18-16-16S15.18 8 24 8s16 7.18 16 16-7.18 16-16 16z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-circle-inner-circle">\n      <path d="M24 4C12.97 4 4 12.97 4 24s8.97 20 20 20 20-8.97 20-20S35.03 4 24 4zm0 36c-8.82 0-16-7.18-16-16S15.18 8 24 8s16 7.18 16 16-7.18 16-16 16zm6-16c0 3.31-2.69 6-6 6s-6-2.69-6-6 2.69-6 6-6 6 2.69 6 6z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-cancel">\n      <path d="M24 4C12.95 4 4 12.95 4 24s8.95 20 20 20 20-8.95 20-20S35.05 4 24 4zm10 27.17L31.17 34 24 26.83 16.83 34 14 31.17 21.17 24 14 16.83 16.83 14 24 21.17 31.17 14 34 16.83 26.83 24 34 31.17z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-replay">\n      <path d="M24 10V2L14 12l10 10v-8c6.63 0 12 5.37 12 12s-5.37 12-12 12-12-5.37-12-12H8c0 8.84 7.16 16 16 16s16-7.16 16-16-7.16-16-16-16z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-repeat">\n      <path d="M14 14h20v6l8-8-8-8v6H10v12h4v-8zm20 20H14v-6l-8 8 8 8v-6h24V26h-4v8z"></path>\n    </symbol>\n    <symbol viewBox="0 96 48 48" id="vjs-icon-replay-5">\n      <path d="M17.689 98l-8.697 8.696 8.697 8.697 2.486-2.485-4.32-4.319h1.302c4.93 0 9.071 1.722 12.424 5.165 3.352 3.443 5.029 7.638 5.029 12.584h3.55c0-2.958-.553-5.73-1.658-8.313-1.104-2.583-2.622-4.841-4.555-6.774-1.932-1.932-4.19-3.45-6.773-4.555-2.584-1.104-5.355-1.657-8.313-1.657H15.5l4.615-4.615zm-8.08 21.659v13.861h11.357v5.008H9.609V143h12.7c.834 0 1.55-.298 2.146-.894.596-.597.895-1.31.895-2.145v-7.781c0-.835-.299-1.55-.895-2.147a2.929 2.929 0 0 0-2.147-.894h-8.227v-5.096H25.35v-4.384z"></path>\n    </symbol>\n    <symbol viewBox="0 96 48 48" id="vjs-icon-replay-10">\n      <path d="M42.315 125.63c0-4.997-1.694-9.235-5.08-12.713-3.388-3.479-7.571-5.218-12.552-5.218h-1.315l4.363 4.363-2.51 2.51-8.787-8.786L25.221 97l2.45 2.45-4.662 4.663h1.375c2.988 0 5.788.557 8.397 1.673 2.61 1.116 4.892 2.65 6.844 4.602 1.953 1.953 3.487 4.234 4.602 6.844 1.116 2.61 1.674 5.41 1.674 8.398zM8.183 142v-19.657H3.176V117.8h9.643V142zm13.63 0c-1.156 0-2.127-.393-2.912-1.178-.778-.778-1.168-1.746-1.168-2.902v-16.04c0-1.156.393-2.127 1.178-2.912.779-.779 1.746-1.168 2.902-1.168h7.696c1.156 0 2.126.392 2.911 1.177.779.78 1.168 1.747 1.168 2.903v16.04c0 1.156-.392 2.127-1.177 2.912-.779.779-1.746 1.168-2.902 1.168zm.556-4.636h6.583v-15.02H22.37z"></path>\n    </symbol>\n    <symbol viewBox="0 96 48 48" id="vjs-icon-replay-30">\n      <path d="M26.047 97l-8.733 8.732 8.733 8.733 2.496-2.494-4.336-4.338h1.307c4.95 0 9.108 1.73 12.474 5.187 3.367 3.458 5.051 7.668 5.051 12.635h3.565c0-2.97-.556-5.751-1.665-8.346-1.109-2.594-2.633-4.862-4.574-6.802-1.94-1.941-4.208-3.466-6.803-4.575-2.594-1.109-5.375-1.664-8.345-1.664H23.85l4.634-4.634zM2.555 117.531v4.688h10.297v5.25H5.873v4.687h6.979v5.156H2.555V142H13.36c1.061 0 1.95-.395 2.668-1.186.718-.79 1.076-1.772 1.076-2.94v-16.218c0-1.168-.358-2.149-1.076-2.94-.717-.79-1.607-1.185-2.668-1.185zm22.482.14c-1.149 0-2.11.39-2.885 1.165-.78.78-1.172 1.744-1.172 2.893v15.943c0 1.149.388 2.11 1.163 2.885.78.78 1.745 1.172 2.894 1.172h7.649c1.148 0 2.11-.388 2.884-1.163.78-.78 1.17-1.745 1.17-2.894v-15.943c0-1.15-.386-2.111-1.16-2.885-.78-.78-1.746-1.172-2.894-1.172zm.553 4.518h6.545v14.93H25.59z"></path>\n    </symbol>\n    <symbol viewBox="0 96 48 48" id="vjs-icon-forward-5">\n      <path d="M29.508 97l-2.431 2.43 4.625 4.625h-1.364c-2.965 0-5.742.554-8.332 1.66-2.589 1.107-4.851 2.629-6.788 4.566-1.937 1.937-3.458 4.2-4.565 6.788-1.107 2.59-1.66 5.367-1.66 8.331h3.557c0-4.957 1.68-9.16 5.04-12.611 3.36-3.45 7.51-5.177 12.451-5.177h1.304l-4.326 4.33 2.49 2.49 8.715-8.716zm-9.783 21.61v13.89h11.382v5.018H19.725V142h12.727a2.93 2.93 0 0 0 2.15-.896 2.93 2.93 0 0 0 .896-2.15v-7.798c0-.837-.299-1.554-.896-2.152a2.93 2.93 0 0 0-2.15-.896h-8.245V123h11.29v-4.392z"></path>\n    </symbol>\n    <symbol viewBox="0 96 48 48" id="vjs-icon-forward-10">\n      <path d="M23.119 97l-2.386 2.383 4.538 4.538h-1.339c-2.908 0-5.633.543-8.173 1.63-2.54 1.085-4.76 2.577-6.66 4.478-1.9 1.9-3.392 4.12-4.478 6.66-1.085 2.54-1.629 5.264-1.629 8.172h3.49c0-4.863 1.648-8.986 4.944-12.372 3.297-3.385 7.368-5.078 12.216-5.078h1.279l-4.245 4.247 2.443 2.442 8.55-8.55zm-9.52 21.45v4.42h4.871V142h4.513v-23.55zm18.136 0c-1.125 0-2.066.377-2.824 1.135-.764.764-1.148 1.709-1.148 2.834v15.612c0 1.124.38 2.066 1.139 2.824.764.764 1.708 1.145 2.833 1.145h7.489c1.125 0 2.066-.378 2.824-1.136.764-.764 1.145-1.709 1.145-2.833v-15.612c0-1.125-.378-2.067-1.136-2.825-.764-.764-1.708-1.145-2.833-1.145zm.54 4.42h6.408v14.617h-6.407z"></path>\n    </symbol>\n    <symbol viewBox="0 96 48 48" id="vjs-icon-forward-30">\n      <path d="M25.549 97l-2.437 2.434 4.634 4.635H26.38c-2.97 0-5.753.555-8.347 1.664-2.594 1.109-4.861 2.633-6.802 4.574-1.94 1.94-3.465 4.207-4.574 6.802-1.109 2.594-1.664 5.377-1.664 8.347h3.565c0-4.967 1.683-9.178 5.05-12.636 3.366-3.458 7.525-5.187 12.475-5.187h1.307l-4.335 4.338 2.495 2.494 8.732-8.732zm-11.553 20.53v4.689h10.297v5.249h-6.978v4.688h6.978v5.156H13.996V142h10.808c1.06 0 1.948-.395 2.666-1.186.718-.79 1.077-1.771 1.077-2.94v-16.217c0-1.169-.36-2.15-1.077-2.94-.718-.79-1.605-1.186-2.666-1.186zm21.174.168c-1.149 0-2.11.389-2.884 1.163-.78.78-1.172 1.745-1.172 2.894v15.942c0 1.15.388 2.11 1.162 2.885.78.78 1.745 1.17 2.894 1.17h7.649c1.149 0 2.11-.386 2.885-1.16.78-.78 1.17-1.746 1.17-2.895v-15.942c0-1.15-.387-2.11-1.161-2.885-.78-.78-1.745-1.172-2.894-1.172zm.552 4.516h6.542v14.931h-6.542z"></path>\n    </symbol>\n    <symbol viewBox="0 0 512 512" id="vjs-icon-audio-description">\n      <g fill-rule="evenodd"><path d="M227.29 381.351V162.993c50.38-1.017 89.108-3.028 117.631 17.126 27.374 19.342 48.734 56.965 44.89 105.325-4.067 51.155-41.335 94.139-89.776 98.475-24.085 2.155-71.972 0-71.972 0s-.84-1.352-.773-2.568m48.755-54.804c31.43 1.26 53.208-16.633 56.495-45.386 4.403-38.51-21.188-63.552-58.041-60.796v103.612c-.036 1.466.575 2.22 1.546 2.57"></path><path d="M383.78 381.328c13.336 3.71 17.387-11.06 23.215-21.408 12.722-22.571 22.294-51.594 22.445-84.774.221-47.594-18.343-82.517-35.6-106.182h-8.51c-.587 3.874 2.226 7.315 3.865 10.276 13.166 23.762 25.367 56.553 25.54 94.194.2 43.176-14.162 79.278-30.955 107.894"></path><path d="M425.154 381.328c13.336 3.71 17.384-11.061 23.215-21.408 12.721-22.571 22.291-51.594 22.445-84.774.221-47.594-18.343-82.517-35.6-106.182h-8.511c-.586 3.874 2.226 7.315 3.866 10.276 13.166 23.762 25.367 56.553 25.54 94.194.2 43.176-14.162 79.278-30.955 107.894"></path><path d="M466.26 381.328c13.337 3.71 17.385-11.061 23.216-21.408 12.722-22.571 22.292-51.594 22.445-84.774.221-47.594-18.343-82.517-35.6-106.182h-8.51c-.587 3.874 2.225 7.315 3.865 10.276 13.166 23.762 25.367 56.553 25.54 94.194.2 43.176-14.162 79.278-30.955 107.894M4.477 383.005H72.58l18.573-28.484 64.169-.135s.065 19.413.065 28.62h48.756V160.307h-58.816c-5.653 9.537-140.85 222.697-140.85 222.697zm152.667-145.282v71.158l-40.453-.27 40.453-70.888z"></path></g>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-next-item">\n      <path d="M12 36l17-12-17-12v24zm20-24v24h4V12h-4z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-previous-item">\n      <path d="M12 12h4v24h-4zm7 12l17 12V12z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-shuffle">\n      <path d="M21.17 18.34L10.83 8 8 10.83l10.34 10.34 2.83-2.83zM29 8l4.09 4.09L8 37.17 10.83 40l25.09-25.09L40 19V8H29zm.66 18.83l-2.83 2.83 6.26 6.26L29 40h11V29l-4.09 4.09-6.25-6.26z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-cast">\n      <path d="M42 6H6c-2.21 0-4 1.79-4 4v6h4v-6h36v28H28v4h14c2.21 0 4-1.79 4-4V10c0-2.21-1.79-4-4-4zM2 36v6h6c0-3.31-2.69-6-6-6zm0-8v4c5.52 0 10 4.48 10 10h4c0-7.73-6.27-14-14-14zm0-8v4c9.94 0 18 8.06 18 18h4c0-12.15-9.85-22-22-22z"></path>\n    </symbol>\n    <symbol viewBox="0 0 48 48" id="vjs-icon-picture-in-picture-enter">\n      <path d="M38 22H22v11.99h16V22zm8 16V9.96C46 7.76 44.2 6 42 6H6C3.8 6 2 7.76 2 9.96V38c0 2.2 1.8 4 4 4h36c2.2 0 4-1.8 4-4zm-4 .04H6V9.94h36v28.1z"></path>\n    </symbol>\n    <symbol viewBox="0 0 22 18" id="vjs-icon-picture-in-picture-exit">\n      <path d="M18 4H4v10h14V4zm4 12V1.98C22 .88 21.1 0 20 0H2C.9 0 0 .88 0 1.98V16c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2zm-2 .02H2V1.97h18v14.05z"></path>\n      <path fill="none" d="M-1-3h24v24H-1z"></path>\n    </symbol>\n    <symbol viewBox="0 0 1792 1792" id="vjs-icon-facebook">\n      <path d="M1343 12v264h-157q-86 0-116 36t-30 108v189h293l-39 296h-254v759H734V905H479V609h255V391q0-186 104-288.5T1115 0q147 0 228 12z"></path>\n    </symbol>\n    <symbol viewBox="0 0 1792 1792" id="vjs-icon-linkedin">\n      <path d="M477 625v991H147V625h330zm21-306q1 73-50.5 122T312 490h-2q-82 0-132-49t-50-122q0-74 51.5-122.5T314 148t133 48.5T498 319zm1166 729v568h-329v-530q0-105-40.5-164.5T1168 862q-63 0-105.5 34.5T999 982q-11 30-11 81v553H659q2-399 2-647t-1-296l-1-48h329v144h-2q20-32 41-56t56.5-52 87-43.5T1285 602q171 0 275 113.5t104 332.5z"></path>\n    </symbol>\n    <symbol viewBox="0 0 1792 1792" id="vjs-icon-twitter">\n      <path d="M1684 408q-67 98-162 167 1 14 1 42 0 130-38 259.5T1369.5 1125 1185 1335.5t-258 146-323 54.5q-271 0-496-145 35 4 78 4 225 0 401-138-105-2-188-64.5T285 1033q33 5 61 5 43 0 85-11-112-23-185.5-111.5T172 710v-4q68 38 146 41-66-44-105-115t-39-154q0-88 44-163 121 149 294.5 238.5T884 653q-8-38-8-74 0-134 94.5-228.5T1199 256q140 0 236 102 109-21 205-78-37 115-142 178 93-10 186-50z"></path>\n    </symbol>\n    <symbol viewBox="0 0 1792 1792" id="vjs-icon-tumblr">\n      <path d="M1328 1329l80 237q-23 35-111 66t-177 32q-104 2-190.5-26T787 1564t-95-106-55.5-120-16.5-118V676H452V461q72-26 129-69.5t91-90 58-102 34-99T779 12q1-5 4.5-8.5T791 0h244v424h333v252h-334v518q0 30 6.5 56t22.5 52.5 49.5 41.5 81.5 14q78-2 134-29z"></path>\n    </symbol>\n    <symbol viewBox="0 0 1792 1792" id="vjs-icon-pinterest">\n      <path d="M1664 896q0 209-103 385.5T1281.5 1561 896 1664q-111 0-218-32 59-93 78-164 9-34 54-211 20 39 73 67.5t114 28.5q121 0 216-68.5t147-188.5 52-270q0-114-59.5-214T1180 449t-255-63q-105 0-196 29t-154.5 77-109 110.5-67 129.5T377 866q0 104 40 183t117 111q30 12 38-20 2-7 8-31t8-30q6-23-11-43-51-61-51-151 0-151 104.5-259.5T904 517q151 0 235.5 82t84.5 213q0 170-68.5 289T980 1220q-61 0-98-43.5T859 1072q8-35 26.5-93.5t30-103T927 800q0-50-27-83t-77-33q-62 0-105 57t-43 142q0 73 25 122l-99 418q-17 70-13 177-206-91-333-281T128 896q0-209 103-385.5T510.5 231 896 128t385.5 103T1561 510.5 1664 896z"></path>\n    </symbol>\n  </defs>\n</svg>', "image/svg+xml");
                    if (e.querySelector("parsererror"))
                        U.warn("Failed to load SVG Icons. Falling back to Font Icons."),
                        this.options_.experimentalSvgIcons = null;
                    else {
                        const t = e.documentElement;
                        t.style.display = "none",
                        this.el_.appendChild(t),
                        this.addClass("vjs-svg-icons-enabled")
                    }
                }
                this.initChildren(),
                this.isAudio("audio" === e.nodeName.toLowerCase()),
                this.controls() ? this.addClass("vjs-controls-enabled") : this.addClass("vjs-controls-disabled"),
                this.el_.setAttribute("role", "region"),
                this.isAudio() ? this.el_.setAttribute("aria-label", this.localize("Audio Player")) : this.el_.setAttribute("aria-label", this.localize("Video Player")),
                this.isAudio() && this.addClass("vjs-audio"),
                he && this.addClass("vjs-touch-enabled"),
                ue || this.addClass("vjs-workinghover"),
                mn.players[this.id_] = this;
                const r = x.split(".")[0];
                this.addClass(`vjs-v${r}`),
                this.userActive(!0),
                this.reportUserActivity(),
                this.one("play", (e=>this.listenForUserActivity_(e))),
                this.on("keydown", (e=>this.handleKeyDown(e))),
                this.on("languagechange", (e=>this.handleLanguagechange(e))),
                this.breakpoints(this.options_.breakpoints),
                this.responsive(this.options_.responsive),
                this.on("ready", (()=>{
                    this.audioPosterMode(this.options_.audioPosterMode),
                    this.audioOnlyMode(this.options_.audioOnlyMode)
                }
                ))
            }
            dispose() {
                this.trigger("dispose"),
                this.off("dispose"),
                ht(a(), this.fsApi_.fullscreenchange, this.boundDocumentFullscreenChange_),
                ht(a(), "keydown", this.boundFullWindowOnEscKey_),
                this.styleEl_ && this.styleEl_.parentNode && (this.styleEl_.parentNode.removeChild(this.styleEl_),
                this.styleEl_ = null),
                mn.players[this.id_] = null,
                this.tag && this.tag.player && (this.tag.player = null),
                this.el_ && this.el_.player && (this.el_.player = null),
                this.tech_ && (this.tech_.dispose(),
                this.isPosterFromTech_ = !1,
                this.poster_ = ""),
                this.playerElIngest_ && (this.playerElIngest_ = null),
                this.tag && (this.tag = null),
                xi[this.id()] = null,
                Ci.names.forEach((e=>{
                    const t = this[Ci[e].getterName]();
                    t && t.off && t.off()
                }
                )),
                super.dispose({
                    restoreEl: this.options_.restoreEl
                })
            }
            createEl() {
                let e, t = this.tag, i = this.playerElIngest_ = t.parentNode && t.parentNode.hasAttribute && t.parentNode.hasAttribute("data-vjs-player");
                const s = "video-js" === this.tag.tagName.toLowerCase();
                i ? e = this.el_ = t.parentNode : s || (e = this.el_ = super.createEl("div"));
                const r = xe(t);
                if (s) {
                    for (e = this.el_ = t,
                    t = this.tag = a().createElement("video"); e.children.length; )
                        t.appendChild(e.firstChild);
                    Se(e, "video-js") || ke(e, "video-js"),
                    e.appendChild(t),
                    i = this.playerElIngest_ = e,
                    Object.keys(e).forEach((i=>{
                        try {
                            t[i] = e[i]
                        } catch (s) {}
                    }
                    ))
                }
                if (t.setAttribute("tabindex", "-1"),
                r.tabindex = "-1",
                te && ae && (t.setAttribute("role", "application"),
                r.role = "application"),
                t.removeAttribute("width"),
                t.removeAttribute("height"),
                "width"in r && delete r.width,
                "height"in r && delete r.height,
                Object.getOwnPropertyNames(r).forEach((function(i) {
                    s && "class" === i || e.setAttribute(i, r[i]),
                    s && t.setAttribute(i, r[i])
                }
                )),
                t.playerId = t.id,
                t.id += "_html5_api",
                t.className = "vjs-tech",
                t.player = e.player = this,
                this.addClass("vjs-paused"),
                !0 !== n().VIDEOJS_NO_DYNAMIC_STYLE) {
                    this.styleEl_ = Je("vjs-styles-dimensions");
                    const e = qe(".vjs-styles-defaults")
                      , t = qe("head");
                    t.insertBefore(this.styleEl_, e ? e.nextSibling : t.firstChild)
                }
                this.fill_ = !1,
                this.fluid_ = !1,
                this.width(this.options_.width),
                this.height(this.options_.height),
                this.fill(this.options_.fill),
                this.fluid(this.options_.fluid),
                this.aspectRatio(this.options_.aspectRatio),
                this.crossOrigin(this.options_.crossOrigin || this.options_.crossorigin);
                const o = t.getElementsByTagName("a");
                for (let n = 0; n < o.length; n++) {
                    const e = o.item(n);
                    ke(e, "vjs-hidden"),
                    e.setAttribute("hidden", "hidden")
                }
                return t.initNetworkState_ = t.networkState,
                t.parentNode && !i && t.parentNode.insertBefore(e, t),
                be(t, e),
                this.children_.unshift(t),
                this.el_.setAttribute("lang", this.language_),
                this.el_.setAttribute("translate", "no"),
                this.el_ = e,
                e
            }
            crossOrigin(e) {
                if ("undefined" === typeof e)
                    return this.techGet_("crossOrigin");
                null === e || "anonymous" === e || "use-credentials" === e ? (this.techCall_("setCrossOrigin", e),
                this.posterImage && this.posterImage.crossOrigin(e)) : U.warn(`crossOrigin must be null,  "anonymous" or "use-credentials", given "${e}"`)
            }
            width(e) {
                return this.dimension("width", e)
            }
            height(e) {
                return this.dimension("height", e)
            }
            dimension(e, t) {
                const i = e + "_";
                if (void 0 === t)
                    return this[i] || 0;
                if ("" === t || "auto" === t)
                    return this[i] = void 0,
                    void this.updateStyleEl_();
                const s = parseFloat(t);
                isNaN(s) ? U.error(`Improper value "${t}" supplied for for ${e}`) : (this[i] = s,
                this.updateStyleEl_())
            }
            fluid(e) {
                if (void 0 === e)
                    return !!this.fluid_;
                var t, i;
                this.fluid_ = !!e,
                St(this) && this.off(["playerreset", "resize"], this.boundUpdateStyleEl_),
                e ? (this.addClass("vjs-fluid"),
                this.fill(!1),
                i = ()=>{
                    this.on(["playerreset", "resize"], this.boundUpdateStyleEl_)
                }
                ,
                St(t = this) ? i() : (t.eventedCallbacks || (t.eventedCallbacks = []),
                t.eventedCallbacks.push(i))) : this.removeClass("vjs-fluid"),
                this.updateStyleEl_()
            }
            fill(e) {
                if (void 0 === e)
                    return !!this.fill_;
                this.fill_ = !!e,
                e ? (this.addClass("vjs-fill"),
                this.fluid(!1)) : this.removeClass("vjs-fill")
            }
            aspectRatio(e) {
                if (void 0 === e)
                    return this.aspectRatio_;
                if (!/^\d+\:\d+$/.test(e))
                    throw new Error("Improper value supplied for aspect ratio. The format should be width:height, for example 16:9.");
                this.aspectRatio_ = e,
                this.fluid(!0),
                this.updateStyleEl_()
            }
            updateStyleEl_() {
                if (!0 === n().VIDEOJS_NO_DYNAMIC_STYLE) {
                    const e = "number" === typeof this.width_ ? this.width_ : this.options_.width
                      , t = "number" === typeof this.height_ ? this.height_ : this.options_.height
                      , i = this.tech_ && this.tech_.el();
                    return void (i && (e >= 0 && (i.width = e),
                    t >= 0 && (i.height = t)))
                }
                let e, t, i, s;
                i = void 0 !== this.aspectRatio_ && "auto" !== this.aspectRatio_ ? this.aspectRatio_ : this.videoWidth() > 0 ? this.videoWidth() + ":" + this.videoHeight() : "16:9";
                const r = i.split(":")
                  , a = r[1] / r[0];
                e = void 0 !== this.width_ ? this.width_ : void 0 !== this.height_ ? this.height_ / a : this.videoWidth() || 300,
                t = void 0 !== this.height_ ? this.height_ : e * a,
                s = /^[^a-zA-Z]/.test(this.id()) ? "dimensions-" + this.id() : this.id() + "-dimensions",
                this.addClass(s),
                Ze(this.styleEl_, `\n      .${s} {\n        width: ${e}px;\n        height: ${t}px;\n      }\n\n      .${s}.vjs-fluid:not(.vjs-audio-only-mode) {\n        padding-top: ${100 * a}%;\n      }\n    `)
            }
            loadTech_(e, t) {
                this.tech_ && this.unloadTech_();
                const i = Mt(e)
                  , s = e.charAt(0).toLowerCase() + e.slice(1);
                "Html5" !== i && this.tag && (Ei.getTech("Html5").disposeMediaElement(this.tag),
                this.tag.player = null,
                this.tag = null),
                this.techName_ = i,
                this.isReady_ = !1;
                let n = this.autoplay();
                ("string" === typeof this.autoplay() || !0 === this.autoplay() && this.options_.normalizeAutoplay) && (n = !1);
                const r = {
                    source: t,
                    autoplay: n,
                    nativeControlsForTouch: this.options_.nativeControlsForTouch,
                    playerId: this.id(),
                    techId: `${this.id()}_${s}_api`,
                    playsinline: this.options_.playsinline,
                    preload: this.options_.preload,
                    loop: this.options_.loop,
                    disablePictureInPicture: this.options_.disablePictureInPicture,
                    muted: this.options_.muted,
                    poster: this.poster(),
                    language: this.language(),
                    playerElIngest: this.playerElIngest_ || !1,
                    "vtt.js": this.options_["vtt.js"],
                    canOverridePoster: !!this.options_.techCanOverridePoster,
                    enableSourceset: this.options_.enableSourceset
                };
                Ci.names.forEach((e=>{
                    const t = Ci[e];
                    r[t.getterName] = this[t.privateName]
                }
                )),
                Object.assign(r, this.options_[i]),
                Object.assign(r, this.options_[s]),
                Object.assign(r, this.options_[e.toLowerCase()]),
                this.tag && (r.tag = this.tag),
                t && t.src === this.cache_.src && this.cache_.currentTime > 0 && (r.startTime = this.cache_.currentTime);
                const a = Ei.getTech(e);
                if (!a)
                    throw new Error(`No Tech named '${i}' exists! '${i}' should be registered using videojs.registerTech()'`);
                this.tech_ = new a(r),
                this.tech_.ready(gt(this, this.handleTechReady_), !0),
                Zt(this.textTracksJson_ || [], this.tech_),
                hn.forEach((e=>{
                    this.on(this.tech_, e, (t=>this[`handleTech${Mt(e)}_`](t)))
                }
                )),
                Object.keys(dn).forEach((e=>{
                    this.on(this.tech_, e, (t=>{
                        0 === this.tech_.playbackRate() && this.tech_.seeking() ? this.queuedCallbacks_.push({
                            callback: this[`handleTech${dn[e]}_`].bind(this),
                            event: t
                        }) : this[`handleTech${dn[e]}_`](t)
                    }
                    ))
                }
                )),
                this.on(this.tech_, "loadstart", (e=>this.handleTechLoadStart_(e))),
                this.on(this.tech_, "sourceset", (e=>this.handleTechSourceset_(e))),
                this.on(this.tech_, "waiting", (e=>this.handleTechWaiting_(e))),
                this.on(this.tech_, "ended", (e=>this.handleTechEnded_(e))),
                this.on(this.tech_, "seeking", (e=>this.handleTechSeeking_(e))),
                this.on(this.tech_, "play", (e=>this.handleTechPlay_(e))),
                this.on(this.tech_, "pause", (e=>this.handleTechPause_(e))),
                this.on(this.tech_, "durationchange", (e=>this.handleTechDurationChange_(e))),
                this.on(this.tech_, "fullscreenchange", ((e,t)=>this.handleTechFullscreenChange_(e, t))),
                this.on(this.tech_, "fullscreenerror", ((e,t)=>this.handleTechFullscreenError_(e, t))),
                this.on(this.tech_, "enterpictureinpicture", (e=>this.handleTechEnterPictureInPicture_(e))),
                this.on(this.tech_, "leavepictureinpicture", (e=>this.handleTechLeavePictureInPicture_(e))),
                this.on(this.tech_, "error", (e=>this.handleTechError_(e))),
                this.on(this.tech_, "posterchange", (e=>this.handleTechPosterChange_(e))),
                this.on(this.tech_, "textdata", (e=>this.handleTechTextData_(e))),
                this.on(this.tech_, "ratechange", (e=>this.handleTechRateChange_(e))),
                this.on(this.tech_, "loadedmetadata", this.boundUpdateStyleEl_),
                this.usingNativeControls(this.techGet_("controls")),
                this.controls() && !this.usingNativeControls() && this.addTechControlsListeners_(),
                this.tech_.el().parentNode === this.el() || "Html5" === i && this.tag || be(this.tech_.el(), this.el()),
                this.tag && (this.tag.player = null,
                this.tag = null)
            }
            unloadTech_() {
                Ci.names.forEach((e=>{
                    const t = Ci[e];
                    this[t.privateName] = this[t.getterName]()
                }
                )),
                this.textTracksJson_ = Jt(this.tech_),
                this.isReady_ = !1,
                this.tech_.dispose(),
                this.tech_ = !1,
                this.isPosterFromTech_ && (this.poster_ = "",
                this.trigger("posterchange")),
                this.isPosterFromTech_ = !1
            }
            tech(e) {
                return void 0 === e && U.warn("Using the tech directly can be dangerous. I hope you know what you're doing.\nSee https://github.com/videojs/video.js/issues/2617 for more info.\n"),
                this.tech_
            }
            version() {
                return {
                    "video.js": x
                }
            }
            addTechControlsListeners_() {
                this.removeTechControlsListeners_(),
                this.on(this.tech_, "click", this.boundHandleTechClick_),
                this.on(this.tech_, "dblclick", this.boundHandleTechDoubleClick_),
                this.on(this.tech_, "touchstart", this.boundHandleTechTouchStart_),
                this.on(this.tech_, "touchmove", this.boundHandleTechTouchMove_),
                this.on(this.tech_, "touchend", this.boundHandleTechTouchEnd_),
                this.on(this.tech_, "tap", this.boundHandleTechTap_)
            }
            removeTechControlsListeners_() {
                this.off(this.tech_, "tap", this.boundHandleTechTap_),
                this.off(this.tech_, "touchstart", this.boundHandleTechTouchStart_),
                this.off(this.tech_, "touchmove", this.boundHandleTechTouchMove_),
                this.off(this.tech_, "touchend", this.boundHandleTechTouchEnd_),
                this.off(this.tech_, "click", this.boundHandleTechClick_),
                this.off(this.tech_, "dblclick", this.boundHandleTechDoubleClick_)
            }
            handleTechReady_() {
                this.triggerReady(),
                this.cache_.volume && this.techCall_("setVolume", this.cache_.volume),
                this.handleTechPosterChange_(),
                this.handleTechDurationChange_()
            }
            handleTechLoadStart_() {
                this.removeClass("vjs-ended", "vjs-seeking"),
                this.error(null),
                this.handleTechDurationChange_(),
                this.paused() ? (this.hasStarted(!1),
                this.trigger("loadstart")) : this.trigger("loadstart"),
                this.manualAutoplay_(!0 === this.autoplay() && this.options_.normalizeAutoplay ? "play" : this.autoplay())
            }
            manualAutoplay_(e) {
                if (!this.tech_ || "string" !== typeof e)
                    return;
                const t = ()=>{
                    const e = this.muted();
                    this.muted(!0);
                    const t = ()=>{
                        this.muted(e)
                    }
                    ;
                    this.playTerminatedQueue_.push(t);
                    const i = this.play();
                    if (Qt(i))
                        return i.catch((e=>{
                            throw t(),
                            new Error(`Rejection at manualAutoplay. Restoring muted value. ${e || ""}`)
                        }
                        ))
                }
                ;
                let i;
                return "any" !== e || this.muted() ? i = "muted" !== e || this.muted() ? this.play() : t() : (i = this.play(),
                Qt(i) && (i = i.catch(t))),
                Qt(i) ? i.then((()=>{
                    this.trigger({
                        type: "autoplay-success",
                        autoplay: e
                    })
                }
                )).catch((()=>{
                    this.trigger({
                        type: "autoplay-failure",
                        autoplay: e
                    })
                }
                )) : void 0
            }
            updateSourceCaches_(e="") {
                let t = e
                  , i = "";
                "string" !== typeof t && (t = e.src,
                i = e.type),
                this.cache_.source = this.cache_.source || {},
                this.cache_.sources = this.cache_.sources || [],
                t && !i && (i = ((e,t)=>{
                    if (!t)
                        return "";
                    if (e.cache_.source.src === t && e.cache_.source.type)
                        return e.cache_.source.type;
                    const i = e.cache_.sources.filter((e=>e.src === t));
                    if (i.length)
                        return i[0].type;
                    const s = e.$$("source");
                    for (let n = 0; n < s.length; n++) {
                        const e = s[n];
                        if (e.type && e.src && e.src === t)
                            return e.type
                    }
                    return Bi(t)
                }
                )(this, t)),
                this.cache_.source = V({}, e, {
                    src: t,
                    type: i
                });
                const s = this.cache_.sources.filter((e=>e.src && e.src === t))
                  , n = []
                  , r = this.$$("source")
                  , a = [];
                for (let o = 0; o < r.length; o++) {
                    const e = xe(r[o]);
                    n.push(e),
                    e.src && e.src === t && a.push(e.src)
                }
                a.length && !s.length ? this.cache_.sources = n : s.length || (this.cache_.sources = [this.cache_.source]),
                this.cache_.src = t
            }
            handleTechSourceset_(e) {
                if (!this.changingSrc_) {
                    let t = e=>this.updateSourceCaches_(e);
                    const i = this.currentSource().src
                      , s = e.src;
                    i && !/^blob:/.test(i) && /^blob:/.test(s) && (!this.lastSource_ || this.lastSource_.tech !== s && this.lastSource_.player !== i) && (t = ()=>{}
                    ),
                    t(s),
                    e.src || this.tech_.any(["sourceset", "loadstart"], (e=>{
                        if ("sourceset" === e.type)
                            return;
                        const t = this.techGet_("currentSrc");
                        this.lastSource_.tech = t,
                        this.updateSourceCaches_(t)
                    }
                    ))
                }
                this.lastSource_ = {
                    player: this.currentSource().src,
                    tech: e.src
                },
                this.trigger({
                    src: e.src,
                    type: "sourceset"
                })
            }
            hasStarted(e) {
                if (void 0 === e)
                    return this.hasStarted_;
                e !== this.hasStarted_ && (this.hasStarted_ = e,
                this.hasStarted_ ? this.addClass("vjs-has-started") : this.removeClass("vjs-has-started"))
            }
            handleTechPlay_() {
                this.removeClass("vjs-ended", "vjs-paused"),
                this.addClass("vjs-playing"),
                this.hasStarted(!0),
                this.trigger("play")
            }
            handleTechRateChange_() {
                this.tech_.playbackRate() > 0 && 0 === this.cache_.lastPlaybackRate && (this.queuedCallbacks_.forEach((e=>e.callback(e.event))),
                this.queuedCallbacks_ = []),
                this.cache_.lastPlaybackRate = this.tech_.playbackRate(),
                this.trigger("ratechange")
            }
            handleTechWaiting_() {
                this.addClass("vjs-waiting"),
                this.trigger("waiting");
                const e = this.currentTime()
                  , t = ()=>{
                    e !== this.currentTime() && (this.removeClass("vjs-waiting"),
                    this.off("timeupdate", t))
                }
                ;
                this.on("timeupdate", t)
            }
            handleTechCanPlay_() {
                this.removeClass("vjs-waiting"),
                this.trigger("canplay")
            }
            handleTechCanPlayThrough_() {
                this.removeClass("vjs-waiting"),
                this.trigger("canplaythrough")
            }
            handleTechPlaying_() {
                this.removeClass("vjs-waiting"),
                this.trigger("playing")
            }
            handleTechSeeking_() {
                this.addClass("vjs-seeking"),
                this.trigger("seeking")
            }
            handleTechSeeked_() {
                this.removeClass("vjs-seeking", "vjs-ended"),
                this.trigger("seeked")
            }
            handleTechPause_() {
                this.removeClass("vjs-playing"),
                this.addClass("vjs-paused"),
                this.trigger("pause")
            }
            handleTechEnded_() {
                this.addClass("vjs-ended"),
                this.removeClass("vjs-waiting"),
                this.options_.loop ? (this.currentTime(0),
                this.play()) : this.paused() || this.pause(),
                this.trigger("ended")
            }
            handleTechDurationChange_() {
                this.duration(this.techGet_("duration"))
            }
            handleTechClick_(e) {
                this.controls_ && (void 0 !== this.options_ && void 0 !== this.options_.userActions && void 0 !== this.options_.userActions.click && !1 === this.options_.userActions.click || (void 0 !== this.options_ && void 0 !== this.options_.userActions && "function" === typeof this.options_.userActions.click ? this.options_.userActions.click.call(this, e) : this.paused() ? Xt(this.play()) : this.pause()))
            }
            handleTechDoubleClick_(e) {
                if (!this.controls_)
                    return;
                Array.prototype.some.call(this.$$(".vjs-control-bar, .vjs-modal-dialog"), (t=>t.contains(e.target))) || void 0 !== this.options_ && void 0 !== this.options_.userActions && void 0 !== this.options_.userActions.doubleClick && !1 === this.options_.userActions.doubleClick || (void 0 !== this.options_ && void 0 !== this.options_.userActions && "function" === typeof this.options_.userActions.doubleClick ? this.options_.userActions.doubleClick.call(this, e) : this.isFullscreen() ? this.exitFullscreen() : this.requestFullscreen())
            }
            handleTechTap_() {
                this.userActive(!this.userActive())
            }
            handleTechTouchStart_() {
                this.userWasActive = this.userActive()
            }
            handleTechTouchMove_() {
                this.userWasActive && this.reportUserActivity()
            }
            handleTechTouchEnd_(e) {
                e.cancelable && e.preventDefault()
            }
            toggleFullscreenClass_() {
                this.isFullscreen() ? this.addClass("vjs-fullscreen") : this.removeClass("vjs-fullscreen")
            }
            documentFullscreenChange_(e) {
                const t = e.target.player;
                if (t && t !== this)
                    return;
                const i = this.el();
                let s = a()[this.fsApi_.fullscreenElement] === i;
                !s && i.matches && (s = i.matches(":" + this.fsApi_.fullscreen)),
                this.isFullscreen(s)
            }
            handleTechFullscreenChange_(e, t) {
                t && (t.nativeIOSFullscreen && (this.addClass("vjs-ios-native-fs"),
                this.tech_.one("webkitendfullscreen", (()=>{
                    this.removeClass("vjs-ios-native-fs")
                }
                ))),
                this.isFullscreen(t.isFullscreen))
            }
            handleTechFullscreenError_(e, t) {
                this.trigger("fullscreenerror", t)
            }
            togglePictureInPictureClass_() {
                this.isInPictureInPicture() ? this.addClass("vjs-picture-in-picture") : this.removeClass("vjs-picture-in-picture")
            }
            handleTechEnterPictureInPicture_(e) {
                this.isInPictureInPicture(!0)
            }
            handleTechLeavePictureInPicture_(e) {
                this.isInPictureInPicture(!1)
            }
            handleTechError_() {
                const e = this.tech_.error();
                e && this.error(e)
            }
            handleTechTextData_() {
                let e = null;
                arguments.length > 1 && (e = arguments[1]),
                this.trigger("textdata", e)
            }
            getCache() {
                return this.cache_
            }
            resetCache_() {
                this.cache_ = {
                    currentTime: 0,
                    initTime: 0,
                    inactivityTimeout: this.options_.inactivityTimeout,
                    duration: NaN,
                    lastVolume: 1,
                    lastPlaybackRate: this.defaultPlaybackRate(),
                    media: null,
                    src: "",
                    source: {},
                    sources: [],
                    playbackRates: [],
                    volume: 1
                }
            }
            techCall_(e, t) {
                this.ready((function() {
                    if (e in Di)
                        return function(e, t, i, s) {
                            return t[i](e.reduce(Mi(i), s))
                        }(this.middleware_, this.tech_, e, t);
                    if (e in Oi)
                        return Ai(this.middleware_, this.tech_, e, t);
                    try {
                        this.tech_ && this.tech_[e](t)
                    } catch (i) {
                        throw U(i),
                        i
                    }
                }
                ), !0)
            }
            techGet_(e) {
                if (this.tech_ && this.tech_.isReady_) {
                    if (e in Li)
                        return function(e, t, i) {
                            return e.reduceRight(Mi(i), t[i]())
                        }(this.middleware_, this.tech_, e);
                    if (e in Oi)
                        return Ai(this.middleware_, this.tech_, e);
                    try {
                        return this.tech_[e]()
                    } catch (t) {
                        if (void 0 === this.tech_[e])
                            throw U(`Video.js: ${e} method not defined for ${this.techName_} playback technology.`, t),
                            t;
                        if ("TypeError" === t.name)
                            throw U(`Video.js: ${e} unavailable on ${this.techName_} playback technology element.`, t),
                            this.tech_.isReady_ = !1,
                            t;
                        throw U(t),
                        t
                    }
                }
            }
            play() {
                return new Promise((e=>{
                    this.play_(e)
                }
                ))
            }
            play_(e=Xt) {
                this.playCallbacks_.push(e);
                const t = Boolean(!this.changingSrc_ && (this.src() || this.currentSrc()))
                  , i = Boolean(ce || ue);
                if (this.waitToPlay_ && (this.off(["ready", "loadstart"], this.waitToPlay_),
                this.waitToPlay_ = null),
                !this.isReady_ || !t)
                    return this.waitToPlay_ = e=>{
                        this.play_()
                    }
                    ,
                    this.one(["ready", "loadstart"], this.waitToPlay_),
                    void (!t && i && this.load());
                const s = this.techGet_("play");
                i && this.hasClass("vjs-ended") && this.resetProgressBar_(),
                null === s ? this.runPlayTerminatedQueue_() : this.runPlayCallbacks_(s)
            }
            runPlayTerminatedQueue_() {
                const e = this.playTerminatedQueue_.slice(0);
                this.playTerminatedQueue_ = [],
                e.forEach((function(e) {
                    e()
                }
                ))
            }
            runPlayCallbacks_(e) {
                const t = this.playCallbacks_.slice(0);
                this.playCallbacks_ = [],
                this.playTerminatedQueue_ = [],
                t.forEach((function(t) {
                    t(e)
                }
                ))
            }
            pause() {
                this.techCall_("pause")
            }
            paused() {
                return !1 !== this.techGet_("paused")
            }
            played() {
                return this.techGet_("played") || jt(0, 0)
            }
            scrubbing(e) {
                if ("undefined" === typeof e)
                    return this.scrubbing_;
                this.scrubbing_ = !!e,
                this.techCall_("setScrubbing", this.scrubbing_),
                e ? this.addClass("vjs-scrubbing") : this.removeClass("vjs-scrubbing")
            }
            currentTime(e) {
                return void 0 === e ? (this.cache_.currentTime = this.techGet_("currentTime") || 0,
                this.cache_.currentTime) : (e < 0 && (e = 0),
                this.isReady_ && !this.changingSrc_ && this.tech_ && this.tech_.isReady_ ? (this.techCall_("setCurrentTime", e),
                this.cache_.initTime = 0,
                void (isFinite(e) && (this.cache_.currentTime = Number(e)))) : (this.cache_.initTime = e,
                this.off("canplay", this.boundApplyInitTime_),
                void this.one("canplay", this.boundApplyInitTime_)))
            }
            applyInitTime_() {
                this.currentTime(this.cache_.initTime)
            }
            duration(e) {
                if (void 0 === e)
                    return void 0 !== this.cache_.duration ? this.cache_.duration : NaN;
                (e = parseFloat(e)) < 0 && (e = 1 / 0),
                e !== this.cache_.duration && (this.cache_.duration = e,
                e === 1 / 0 ? this.addClass("vjs-live") : this.removeClass("vjs-live"),
                isNaN(e) || this.trigger("durationchange"))
            }
            remainingTime() {
                return this.duration() - this.currentTime()
            }
            remainingTimeDisplay() {
                return Math.floor(this.duration()) - Math.floor(this.currentTime())
            }
            buffered() {
                let e = this.techGet_("buffered");
                return e && e.length || (e = jt(0, 0)),
                e
            }
            seekable() {
                let e = this.techGet_("seekable");
                return e && e.length || (e = jt(0, 0)),
                e
            }
            seeking() {
                return this.techGet_("seeking")
            }
            ended() {
                return this.techGet_("ended")
            }
            networkState() {
                return this.techGet_("networkState")
            }
            readyState() {
                return this.techGet_("readyState")
            }
            bufferedPercent() {
                return Gt(this.buffered(), this.duration())
            }
            bufferedEnd() {
                const e = this.buffered()
                  , t = this.duration();
                let i = e.end(e.length - 1);
                return i > t && (i = t),
                i
            }
            volume(e) {
                let t;
                return void 0 !== e ? (t = Math.max(0, Math.min(1, e)),
                this.cache_.volume = t,
                this.techCall_("setVolume", t),
                void (t > 0 && this.lastVolume_(t))) : (t = parseFloat(this.techGet_("volume")),
                isNaN(t) ? 1 : t)
            }
            muted(e) {
                if (void 0 === e)
                    return this.techGet_("muted") || !1;
                this.techCall_("setMuted", e)
            }
            defaultMuted(e) {
                return void 0 !== e && this.techCall_("setDefaultMuted", e),
                this.techGet_("defaultMuted") || !1
            }
            lastVolume_(e) {
                if (void 0 === e || 0 === e)
                    return this.cache_.lastVolume;
                this.cache_.lastVolume = e
            }
            supportsFullScreen() {
                return this.techGet_("supportsFullScreen") || !1
            }
            isFullscreen(e) {
                if (void 0 !== e) {
                    const t = this.isFullscreen_;
                    return this.isFullscreen_ = Boolean(e),
                    this.isFullscreen_ !== t && this.fsApi_.prefixed && this.trigger("fullscreenchange"),
                    void this.toggleFullscreenClass_()
                }
                return this.isFullscreen_
            }
            requestFullscreen(e) {
                this.isInPictureInPicture() && this.exitPictureInPicture();
                const t = this;
                return new Promise(((i,s)=>{
                    function n() {
                        t.off("fullscreenerror", a),
                        t.off("fullscreenchange", r)
                    }
                    function r() {
                        n(),
                        i()
                    }
                    function a(e, t) {
                        n(),
                        s(t)
                    }
                    t.one("fullscreenchange", r),
                    t.one("fullscreenerror", a);
                    const o = t.requestFullscreenHelper_(e);
                    o && (o.then(n, n),
                    o.then(i, s))
                }
                ))
            }
            requestFullscreenHelper_(e) {
                let t;
                if (this.fsApi_.prefixed || (t = this.options_.fullscreen && this.options_.fullscreen.options || {},
                void 0 !== e && (t = e)),
                this.fsApi_.requestFullscreen) {
                    const e = this.el_[this.fsApi_.requestFullscreen](t);
                    return e && e.then((()=>this.isFullscreen(!0)), (()=>this.isFullscreen(!1))),
                    e
                }
                this.tech_.supportsFullScreen() && !0 === !this.options_.preferFullWindow ? this.techCall_("enterFullScreen") : this.enterFullWindow()
            }
            exitFullscreen() {
                const e = this;
                return new Promise(((t,i)=>{
                    function s() {
                        e.off("fullscreenerror", r),
                        e.off("fullscreenchange", n)
                    }
                    function n() {
                        s(),
                        t()
                    }
                    function r(e, t) {
                        s(),
                        i(t)
                    }
                    e.one("fullscreenchange", n),
                    e.one("fullscreenerror", r);
                    const a = e.exitFullscreenHelper_();
                    a && (a.then(s, s),
                    a.then(t, i))
                }
                ))
            }
            exitFullscreenHelper_() {
                if (this.fsApi_.requestFullscreen) {
                    const e = a()[this.fsApi_.exitFullscreen]();
                    return e && Xt(e.then((()=>this.isFullscreen(!1)))),
                    e
                }
                this.tech_.supportsFullScreen() && !0 === !this.options_.preferFullWindow ? this.techCall_("exitFullScreen") : this.exitFullWindow()
            }
            enterFullWindow() {
                this.isFullscreen(!0),
                this.isFullWindow = !0,
                this.docOrigOverflow = a().documentElement.style.overflow,
                lt(a(), "keydown", this.boundFullWindowOnEscKey_),
                a().documentElement.style.overflow = "hidden",
                ke(a().body, "vjs-full-window"),
                this.trigger("enterFullWindow")
            }
            fullWindowOnEscKey(e) {
                l().isEventKey(e, "Esc") && !0 === this.isFullscreen() && (this.isFullWindow ? this.exitFullWindow() : this.exitFullscreen())
            }
            exitFullWindow() {
                this.isFullscreen(!1),
                this.isFullWindow = !1,
                ht(a(), "keydown", this.boundFullWindowOnEscKey_),
                a().documentElement.style.overflow = this.docOrigOverflow,
                Ce(a().body, "vjs-full-window"),
                this.trigger("exitFullWindow")
            }
            disablePictureInPicture(e) {
                if (void 0 === e)
                    return this.techGet_("disablePictureInPicture");
                this.techCall_("setDisablePictureInPicture", e),
                this.options_.disablePictureInPicture = e,
                this.trigger("disablepictureinpicturechanged")
            }
            isInPictureInPicture(e) {
                return void 0 !== e ? (this.isInPictureInPicture_ = !!e,
                void this.togglePictureInPictureClass_()) : !!this.isInPictureInPicture_
            }
            requestPictureInPicture() {
                if (this.options_.enableDocumentPictureInPicture && n().documentPictureInPicture) {
                    const e = a().createElement(this.el().tagName);
                    return e.classList = this.el().classList,
                    e.classList.add("vjs-pip-container"),
                    this.posterImage && e.appendChild(this.posterImage.el().cloneNode(!0)),
                    this.titleBar && e.appendChild(this.titleBar.el().cloneNode(!0)),
                    e.appendChild(ve("p", {
                        className: "vjs-pip-text"
                    }, {}, this.localize("Playing in picture-in-picture"))),
                    n().documentPictureInPicture.requestWindow({
                        width: this.videoWidth(),
                        height: this.videoHeight()
                    }).then((t=>(ze(t),
                    this.el_.parentNode.insertBefore(e, this.el_),
                    t.document.body.appendChild(this.el_),
                    t.document.body.classList.add("vjs-pip-window"),
                    this.player_.isInPictureInPicture(!0),
                    this.player_.trigger("enterpictureinpicture"),
                    t.addEventListener("pagehide", (t=>{
                        const i = t.target.querySelector(".video-js");
                        e.parentNode.replaceChild(i, e),
                        this.player_.isInPictureInPicture(!1),
                        this.player_.trigger("leavepictureinpicture")
                    }
                    )),
                    t)))
                }
                return "pictureInPictureEnabled"in a() && !1 === this.disablePictureInPicture() ? this.techGet_("requestPictureInPicture") : Promise.reject("No PiP mode is available")
            }
            exitPictureInPicture() {
                return n().documentPictureInPicture && n().documentPictureInPicture.window ? (n().documentPictureInPicture.window.close(),
                Promise.resolve()) : "pictureInPictureEnabled"in a() ? a().exitPictureInPicture() : void 0
            }
            handleKeyDown(e) {
                const {userActions: t} = this.options_;
                if (!t || !t.hotkeys)
                    return;
                (e=>{
                    const t = e.tagName.toLowerCase();
                    if (e.isContentEditable)
                        return !0;
                    if ("input" === t)
                        return -1 === ["button", "checkbox", "hidden", "radio", "reset", "submit"].indexOf(e.type);
                    return -1 !== ["textarea"].indexOf(t)
                }
                )(this.el_.ownerDocument.activeElement) || ("function" === typeof t.hotkeys ? t.hotkeys.call(this, e) : this.handleHotkeys(e))
            }
            handleHotkeys(e) {
                const t = this.options_.userActions ? this.options_.userActions.hotkeys : {}
                  , {fullscreenKey: i=(e=>l().isEventKey(e, "f")), muteKey: s=(e=>l().isEventKey(e, "m")), playPauseKey: n=(e=>l().isEventKey(e, "k") || l().isEventKey(e, "Space"))} = t;
                if (i.call(this, e)) {
                    e.preventDefault(),
                    e.stopPropagation();
                    const t = Bt.getComponent("FullscreenToggle");
                    !1 !== a()[this.fsApi_.fullscreenEnabled] && t.prototype.handleClick.call(this, e)
                } else if (s.call(this, e)) {
                    e.preventDefault(),
                    e.stopPropagation();
                    Bt.getComponent("MuteToggle").prototype.handleClick.call(this, e)
                } else if (n.call(this, e)) {
                    e.preventDefault(),
                    e.stopPropagation();
                    Bt.getComponent("PlayToggle").prototype.handleClick.call(this, e)
                }
            }
            canPlayType(e) {
                let t;
                for (let i = 0, s = this.options_.techOrder; i < s.length; i++) {
                    const n = s[i];
                    let r = Ei.getTech(n);
                    if (r || (r = Bt.getComponent(n)),
                    r) {
                        if (r.isSupported() && (t = r.canPlayType(e),
                        t))
                            return t
                    } else
                        U.error(`The "${n}" tech is undefined. Skipped browser support check for that tech.`)
                }
                return ""
            }
            selectSource(e) {
                const t = this.options_.techOrder.map((e=>[e, Ei.getTech(e)])).filter((([e,t])=>t ? t.isSupported() : (U.error(`The "${e}" tech is undefined. Skipped browser support check for that tech.`),
                !1)))
                  , i = function(e, t, i) {
                    let s;
                    return e.some((e=>t.some((t=>{
                        if (s = i(e, t),
                        s)
                            return !0
                    }
                    )))),
                    s
                };
                let s;
                const n = ([e,t],i)=>{
                    if (t.canPlaySource(i, this.options_[e.toLowerCase()]))
                        return {
                            source: i,
                            tech: e
                        }
                }
                ;
                var r;
                return s = this.options_.sourceOrder ? i(e, t, (r = n,
                (e,t)=>r(t, e))) : i(t, e, n),
                s || !1
            }
            handleSrc_(e, t) {
                if ("undefined" === typeof e)
                    return this.cache_.src || "";
                this.resetRetryOnError_ && this.resetRetryOnError_();
                const i = Ni(e);
                if (i.length) {
                    if (this.changingSrc_ = !0,
                    t || (this.cache_.sources = i),
                    this.updateSourceCaches_(i[0]),
                    Pi(this, i[0], ((e,s)=>{
                        this.middleware_ = s,
                        t || (this.cache_.sources = i),
                        this.updateSourceCaches_(e);
                        if (this.src_(e))
                            return i.length > 1 ? this.handleSrc_(i.slice(1)) : (this.changingSrc_ = !1,
                            this.setTimeout((function() {
                                this.error({
                                    code: 4,
                                    message: this.options_.notSupportedMessage
                                })
                            }
                            ), 0),
                            void this.triggerReady());
                        var n, r;
                        n = s,
                        r = this.tech_,
                        n.forEach((e=>e.setTech && e.setTech(r)))
                    }
                    )),
                    i.length > 1) {
                        const e = ()=>{
                            this.error(null),
                            this.handleSrc_(i.slice(1), !0)
                        }
                          , t = ()=>{
                            this.off("error", e)
                        }
                        ;
                        this.one("error", e),
                        this.one("playing", t),
                        this.resetRetryOnError_ = ()=>{
                            this.off("error", e),
                            this.off("playing", t)
                        }
                    }
                } else
                    this.setTimeout((function() {
                        this.error({
                            code: 4,
                            message: this.options_.notSupportedMessage
                        })
                    }
                    ), 0)
            }
            src(e) {
                return this.handleSrc_(e, !1)
            }
            src_(e) {
                const t = this.selectSource([e]);
                return !t || (Rt(t.tech, this.techName_) ? (this.ready((function() {
                    this.tech_.constructor.prototype.hasOwnProperty("setSource") ? this.techCall_("setSource", e) : this.techCall_("src", e.src),
                    this.changingSrc_ = !1
                }
                ), !0),
                !1) : (this.changingSrc_ = !0,
                this.loadTech_(t.tech, t.source),
                this.tech_.ready((()=>{
                    this.changingSrc_ = !1
                }
                )),
                !1))
            }
            load() {
                this.tech_ && this.tech_.vhs ? this.src(this.currentSource()) : this.techCall_("load")
            }
            reset() {
                if (this.paused())
                    this.doReset_();
                else {
                    Xt(this.play().then((()=>this.doReset_())))
                }
            }
            doReset_() {
                this.tech_ && this.tech_.clearTracks("text"),
                this.removeClass("vjs-playing"),
                this.addClass("vjs-paused"),
                this.resetCache_(),
                this.poster(""),
                this.loadTech_(this.options_.techOrder[0], null),
                this.techCall_("reset"),
                this.resetControlBarUI_(),
                this.error(null),
                this.titleBar && this.titleBar.update({
                    title: void 0,
                    description: void 0
                }),
                St(this) && this.trigger("playerreset")
            }
            resetControlBarUI_() {
                this.resetProgressBar_(),
                this.resetPlaybackRate_(),
                this.resetVolumeBar_()
            }
            resetProgressBar_() {
                this.currentTime(0);
                const {currentTimeDisplay: e, durationDisplay: t, progressControl: i, remainingTimeDisplay: s} = this.controlBar || {}
                  , {seekBar: n} = i || {};
                e && e.updateContent(),
                t && t.updateContent(),
                s && s.updateContent(),
                n && (n.update(),
                n.loadProgressBar && n.loadProgressBar.update())
            }
            resetPlaybackRate_() {
                this.playbackRate(this.defaultPlaybackRate()),
                this.handleTechRateChange_()
            }
            resetVolumeBar_() {
                this.volume(1),
                this.trigger("volumechange")
            }
            currentSources() {
                const e = this.currentSource()
                  , t = [];
                return 0 !== Object.keys(e).length && t.push(e),
                this.cache_.sources || t
            }
            currentSource() {
                return this.cache_.source || {}
            }
            currentSrc() {
                return this.currentSource() && this.currentSource().src || ""
            }
            currentType() {
                return this.currentSource() && this.currentSource().type || ""
            }
            preload(e) {
                return void 0 !== e ? (this.techCall_("setPreload", e),
                void (this.options_.preload = e)) : this.techGet_("preload")
            }
            autoplay(e) {
                if (void 0 === e)
                    return this.options_.autoplay || !1;
                let t;
                "string" === typeof e && /(any|play|muted)/.test(e) || !0 === e && this.options_.normalizeAutoplay ? (this.options_.autoplay = e,
                this.manualAutoplay_("string" === typeof e ? e : "play"),
                t = !1) : this.options_.autoplay = !!e,
                t = "undefined" === typeof t ? this.options_.autoplay : t,
                this.tech_ && this.techCall_("setAutoplay", t)
            }
            playsinline(e) {
                return void 0 !== e && (this.techCall_("setPlaysinline", e),
                this.options_.playsinline = e),
                this.techGet_("playsinline")
            }
            loop(e) {
                return void 0 !== e ? (this.techCall_("setLoop", e),
                void (this.options_.loop = e)) : this.techGet_("loop")
            }
            poster(e) {
                if (void 0 === e)
                    return this.poster_;
                e || (e = ""),
                e !== this.poster_ && (this.poster_ = e,
                this.techCall_("setPoster", e),
                this.isPosterFromTech_ = !1,
                this.trigger("posterchange"))
            }
            handleTechPosterChange_() {
                if ((!this.poster_ || this.options_.techCanOverridePoster) && this.tech_ && this.tech_.poster) {
                    const e = this.tech_.poster() || "";
                    e !== this.poster_ && (this.poster_ = e,
                    this.isPosterFromTech_ = !0,
                    this.trigger("posterchange"))
                }
            }
            controls(e) {
                if (void 0 === e)
                    return !!this.controls_;
                e = !!e,
                this.controls_ !== e && (this.controls_ = e,
                this.usingNativeControls() && this.techCall_("setControls", e),
                this.controls_ ? (this.removeClass("vjs-controls-disabled"),
                this.addClass("vjs-controls-enabled"),
                this.trigger("controlsenabled"),
                this.usingNativeControls() || this.addTechControlsListeners_()) : (this.removeClass("vjs-controls-enabled"),
                this.addClass("vjs-controls-disabled"),
                this.trigger("controlsdisabled"),
                this.usingNativeControls() || this.removeTechControlsListeners_()))
            }
            usingNativeControls(e) {
                if (void 0 === e)
                    return !!this.usingNativeControls_;
                e = !!e,
                this.usingNativeControls_ !== e && (this.usingNativeControls_ = e,
                this.usingNativeControls_ ? (this.addClass("vjs-using-native-controls"),
                this.trigger("usingnativecontrols")) : (this.removeClass("vjs-using-native-controls"),
                this.trigger("usingcustomcontrols")))
            }
            error(e) {
                if (void 0 === e)
                    return this.error_ || null;
                if (P("beforeerror").forEach((t=>{
                    const i = t(this, e);
                    q(i) && !Array.isArray(i) || "string" === typeof i || "number" === typeof i || null === i ? e = i : this.log.error("please return a value that MediaError expects in beforeerror hooks")
                }
                )),
                this.options_.suppressNotSupportedError && e && 4 === e.code) {
                    const t = function() {
                        this.error(e)
                    };
                    return this.options_.suppressNotSupportedError = !1,
                    this.any(["click", "touchstart"], t),
                    void this.one("loadstart", (function() {
                        this.off(["click", "touchstart"], t)
                    }
                    ))
                }
                if (null === e)
                    return this.error_ = null,
                    this.removeClass("vjs-error"),
                    void (this.errorDisplay && this.errorDisplay.close());
                this.error_ = new Kt(e),
                this.addClass("vjs-error"),
                U.error(`(CODE:${this.error_.code} ${Kt.errorTypes[this.error_.code]})`, this.error_.message, this.error_),
                this.trigger("error"),
                P("error").forEach((e=>e(this, this.error_)))
            }
            reportUserActivity(e) {
                this.userActivity_ = !0
            }
            userActive(e) {
                if (void 0 === e)
                    return this.userActive_;
                if ((e = !!e) !== this.userActive_) {
                    if (this.userActive_ = e,
                    this.userActive_)
                        return this.userActivity_ = !0,
                        this.removeClass("vjs-user-inactive"),
                        this.addClass("vjs-user-active"),
                        void this.trigger("useractive");
                    this.tech_ && this.tech_.one("mousemove", (function(e) {
                        e.stopPropagation(),
                        e.preventDefault()
                    }
                    )),
                    this.userActivity_ = !1,
                    this.removeClass("vjs-user-active"),
                    this.addClass("vjs-user-inactive"),
                    this.trigger("userinactive")
                }
            }
            listenForUserActivity_() {
                let e, t, i;
                const s = gt(this, this.reportUserActivity)
                  , n = function(t) {
                    s(),
                    this.clearInterval(e)
                };
                this.on("mousedown", (function() {
                    s(),
                    this.clearInterval(e),
                    e = this.setInterval(s, 250)
                }
                )),
                this.on("mousemove", (function(e) {
                    e.screenX === t && e.screenY === i || (t = e.screenX,
                    i = e.screenY,
                    s())
                }
                )),
                this.on("mouseup", n),
                this.on("mouseleave", n);
                const r = this.getChild("controlBar");
                let a;
                !r || ue || Y || (r.on("mouseenter", (function(e) {
                    0 !== this.player().options_.inactivityTimeout && (this.player().cache_.inactivityTimeout = this.player().options_.inactivityTimeout),
                    this.player().options_.inactivityTimeout = 0
                }
                )),
                r.on("mouseleave", (function(e) {
                    this.player().options_.inactivityTimeout = this.player().cache_.inactivityTimeout
                }
                ))),
                this.on("keydown", s),
                this.on("keyup", s);
                this.setInterval((function() {
                    if (!this.userActivity_)
                        return;
                    this.userActivity_ = !1,
                    this.userActive(!0),
                    this.clearTimeout(a);
                    const e = this.options_.inactivityTimeout;
                    e <= 0 || (a = this.setTimeout((function() {
                        this.userActivity_ || this.userActive(!1)
                    }
                    ), e))
                }
                ), 250)
            }
            playbackRate(e) {
                if (void 0 === e)
                    return this.tech_ && this.tech_.featuresPlaybackRate ? this.cache_.lastPlaybackRate || this.techGet_("playbackRate") : 1;
                this.techCall_("setPlaybackRate", e)
            }
            defaultPlaybackRate(e) {
                return void 0 !== e ? this.techCall_("setDefaultPlaybackRate", e) : this.tech_ && this.tech_.featuresPlaybackRate ? this.techGet_("defaultPlaybackRate") : 1
            }
            isAudio(e) {
                if (void 0 === e)
                    return !!this.isAudio_;
                this.isAudio_ = !!e
            }
            enableAudioOnlyUI_() {
                this.addClass("vjs-audio-only-mode");
                const e = this.children()
                  , t = this.getChild("ControlBar")
                  , i = t && t.currentHeight();
                e.forEach((e=>{
                    e !== t && e.el_ && !e.hasClass("vjs-hidden") && (e.hide(),
                    this.audioOnlyCache_.hiddenChildren.push(e))
                }
                )),
                this.audioOnlyCache_.playerHeight = this.currentHeight(),
                this.height(i),
                this.trigger("audioonlymodechange")
            }
            disableAudioOnlyUI_() {
                this.removeClass("vjs-audio-only-mode"),
                this.audioOnlyCache_.hiddenChildren.forEach((e=>e.show())),
                this.height(this.audioOnlyCache_.playerHeight),
                this.trigger("audioonlymodechange")
            }
            audioOnlyMode(e) {
                if ("boolean" !== typeof e || e === this.audioOnlyMode_)
                    return this.audioOnlyMode_;
                if (this.audioOnlyMode_ = e,
                e) {
                    const e = [];
                    return this.isInPictureInPicture() && e.push(this.exitPictureInPicture()),
                    this.isFullscreen() && e.push(this.exitFullscreen()),
                    this.audioPosterMode() && e.push(this.audioPosterMode(!1)),
                    Promise.all(e).then((()=>this.enableAudioOnlyUI_()))
                }
                return Promise.resolve().then((()=>this.disableAudioOnlyUI_()))
            }
            enablePosterModeUI_() {
                (this.tech_ && this.tech_).hide(),
                this.addClass("vjs-audio-poster-mode"),
                this.trigger("audiopostermodechange")
            }
            disablePosterModeUI_() {
                (this.tech_ && this.tech_).show(),
                this.removeClass("vjs-audio-poster-mode"),
                this.trigger("audiopostermodechange")
            }
            audioPosterMode(e) {
                if ("boolean" !== typeof e || e === this.audioPosterMode_)
                    return this.audioPosterMode_;
                if (this.audioPosterMode_ = e,
                e) {
                    if (this.audioOnlyMode()) {
                        return this.audioOnlyMode(!1).then((()=>{
                            this.enablePosterModeUI_()
                        }
                        ))
                    }
                    return Promise.resolve().then((()=>{
                        this.enablePosterModeUI_()
                    }
                    ))
                }
                return Promise.resolve().then((()=>{
                    this.disablePosterModeUI_()
                }
                ))
            }
            addTextTrack(e, t, i) {
                if (this.tech_)
                    return this.tech_.addTextTrack(e, t, i)
            }
            addRemoteTextTrack(e, t) {
                if (this.tech_)
                    return this.tech_.addRemoteTextTrack(e, t)
            }
            removeRemoteTextTrack(e={}) {
                let {track: t} = e;
                if (t || (t = e),
                this.tech_)
                    return this.tech_.removeRemoteTextTrack(t)
            }
            getVideoPlaybackQuality() {
                return this.techGet_("getVideoPlaybackQuality")
            }
            videoWidth() {
                return this.tech_ && this.tech_.videoWidth && this.tech_.videoWidth() || 0
            }
            videoHeight() {
                return this.tech_ && this.tech_.videoHeight && this.tech_.videoHeight() || 0
            }
            language(e) {
                if (void 0 === e)
                    return this.language_;
                this.language_ !== String(e).toLowerCase() && (this.language_ = String(e).toLowerCase(),
                St(this) && this.trigger("languagechange"))
            }
            languages() {
                return V(mn.prototype.options_.languages, this.languages_)
            }
            toJSON() {
                const e = V(this.options_)
                  , t = e.tracks;
                e.tracks = [];
                for (let i = 0; i < t.length; i++) {
                    let s = t[i];
                    s = V(s),
                    s.player = void 0,
                    e.tracks[i] = s
                }
                return e
            }
            createModal(e, t) {
                (t = t || {}).content = e || "";
                const i = new ei(this,t);
                return this.addChild(i),
                i.on("dispose", (()=>{
                    this.removeChild(i)
                }
                )),
                i.open(),
                i
            }
            updateCurrentBreakpoint_() {
                if (!this.responsive())
                    return;
                const e = this.currentBreakpoint()
                  , t = this.currentWidth();
                for (let i = 0; i < un.length; i++) {
                    const s = un[i];
                    if (t <= this.breakpoints_[s]) {
                        if (e === s)
                            return;
                        e && this.removeClass(cn[e]),
                        this.addClass(cn[s]),
                        this.breakpoint_ = s;
                        break
                    }
                }
            }
            removeCurrentBreakpoint_() {
                const e = this.currentBreakpointClass();
                this.breakpoint_ = "",
                e && this.removeClass(e)
            }
            breakpoints(e) {
                return void 0 === e || (this.breakpoint_ = "",
                this.breakpoints_ = Object.assign({}, pn, e),
                this.updateCurrentBreakpoint_()),
                Object.assign(this.breakpoints_)
            }
            responsive(e) {
                if (void 0 === e)
                    return this.responsive_;
                return (e = Boolean(e)) !== this.responsive_ ? (this.responsive_ = e,
                e ? (this.on("playerresize", this.boundUpdateCurrentBreakpoint_),
                this.updateCurrentBreakpoint_()) : (this.off("playerresize", this.boundUpdateCurrentBreakpoint_),
                this.removeCurrentBreakpoint_()),
                e) : void 0
            }
            currentBreakpoint() {
                return this.breakpoint_
            }
            currentBreakpointClass() {
                return cn[this.breakpoint_] || ""
            }
            loadMedia(e, t) {
                if (!e || "object" !== typeof e)
                    return;
                const i = this.crossOrigin();
                this.reset(),
                this.cache_.media = V(e);
                const {artist: s, artwork: n, description: r, poster: a, src: o, textTracks: l, title: h} = this.cache_.media;
                !n && a && (this.cache_.media.artwork = [{
                    src: a,
                    type: Bi(a)
                }]),
                i && this.crossOrigin(i),
                o && this.src(o),
                a && this.poster(a),
                Array.isArray(l) && l.forEach((e=>this.addRemoteTextTrack(e, !1))),
                this.titleBar && this.titleBar.update({
                    title: h,
                    description: r || s || ""
                }),
                this.ready(t)
            }
            getMedia() {
                if (!this.cache_.media) {
                    const e = this.poster()
                      , t = {
                        src: this.currentSources(),
                        textTracks: Array.prototype.map.call(this.remoteTextTracks(), (e=>({
                            kind: e.kind,
                            label: e.label,
                            language: e.language,
                            src: e.src
                        })))
                    };
                    return e && (t.poster = e,
                    t.artwork = [{
                        src: t.poster,
                        type: Bi(t.poster)
                    }]),
                    t
                }
                return V(this.cache_.media)
            }
            static getTagSettings(e) {
                const t = {
                    sources: [],
                    tracks: []
                }
                  , i = xe(e)
                  , s = i["data-setup"];
                if (Se(e, "vjs-fill") && (i.fill = !0),
                Se(e, "vjs-fluid") && (i.fluid = !0),
                null !== s) {
                    const [e,t] = d()(s || "{}");
                    e && U.error(e),
                    Object.assign(i, t)
                }
                if (Object.assign(t, i),
                e.hasChildNodes()) {
                    const i = e.childNodes;
                    for (let e = 0, s = i.length; e < s; e++) {
                        const s = i[e]
                          , n = s.nodeName.toLowerCase();
                        "source" === n ? t.sources.push(xe(s)) : "track" === n && t.tracks.push(xe(s))
                    }
                }
                return t
            }
            debug(e) {
                if (void 0 === e)
                    return this.debugEnabled_;
                e ? (this.trigger("debugon"),
                this.previousLogLevel_ = this.log.level,
                this.log.level("debug"),
                this.debugEnabled_ = !0) : (this.trigger("debugoff"),
                this.log.level(this.previousLogLevel_),
                this.previousLogLevel_ = void 0,
                this.debugEnabled_ = !1)
            }
            playbackRates(e) {
                if (void 0 === e)
                    return this.cache_.playbackRates;
                Array.isArray(e) && e.every((e=>"number" === typeof e)) && (this.cache_.playbackRates = e,
                this.trigger("playbackrateschange"))
            }
        }
        Ci.names.forEach((function(e) {
            const t = Ci[e];
            mn.prototype[t.getterName] = function() {
                return this.tech_ ? this.tech_[t.getterName]() : (this[t.privateName] = this[t.privateName] || new t.ListClass,
                this[t.privateName])
            }
        }
        )),
        mn.prototype.crossorigin = mn.prototype.crossOrigin,
        mn.players = {};
        const gn = n().navigator;
        mn.prototype.options_ = {
            techOrder: Ei.defaultTechOrder_,
            html5: {},
            enableSourceset: !0,
            inactivityTimeout: 2e3,
            playbackRates: [],
            liveui: !1,
            children: ["mediaLoader", "posterImage", "titleBar", "textTrackDisplay", "loadingSpinner", "bigPlayButton", "liveTracker", "controlBar", "errorDisplay", "textTrackSettings", "resizeManager"],
            language: gn && (gn.languages && gn.languages[0] || gn.userLanguage || gn.language) || "en",
            languages: {},
            notSupportedMessage: "No compatible source was found for this media.",
            normalizeAutoplay: !1,
            fullscreen: {
                options: {
                    navigationUI: "hide"
                }
            },
            breakpoints: {},
            responsive: !1,
            audioOnlyMode: !1,
            audioPosterMode: !1,
            enableSmoothSeeking: !1
        },
        hn.forEach((function(e) {
            mn.prototype[`handleTech${Mt(e)}_`] = function() {
                return this.trigger(e)
            }
        }
        )),
        Bt.registerComponent("Player", mn);
        const fn = "plugin"
          , _n = {}
          , yn = e=>_n.hasOwnProperty(e)
          , vn = e=>yn(e) ? _n[e] : void 0
          , Tn = (e,t)=>{
            e.activePlugins_ = e.activePlugins_ || {},
            e.activePlugins_[t] = !0
        }
          , bn = (e,t,i)=>{
            const s = (i ? "before" : "") + "pluginsetup";
            e.trigger(s, t),
            e.trigger(s + ":" + t.name, t)
        }
          , Sn = (e,t)=>(t.prototype.name = e,
        function(...i) {
            bn(this, {
                name: e,
                plugin: t,
                instance: null
            }, !0);
            const s = new t(...[this, ...i]);
            return this[e] = ()=>s,
            bn(this, s.getEventHash()),
            s
        }
        );
        class kn {
            constructor(e) {
                if (this.constructor === kn)
                    throw new Error("Plugin must be sub-classed; not directly instantiated.");
                this.player = e,
                this.log || (this.log = this.player.log.createLogger(this.name)),
                At(this),
                delete this.trigger,
                Dt(this, this.constructor.defaultState),
                Tn(e, this.name),
                this.dispose = this.dispose.bind(this),
                e.on("dispose", this.dispose)
            }
            version() {
                return this.constructor.VERSION
            }
            getEventHash(e={}) {
                return e.name = this.name,
                e.plugin = this.constructor,
                e.instance = this,
                e
            }
            trigger(e, t={}) {
                return dt(this.eventBusEl_, e, this.getEventHash(t))
            }
            handleStateChanged(e) {}
            dispose() {
                const {name: e, player: t} = this;
                this.trigger("dispose"),
                this.off(),
                t.off("dispose", this.dispose),
                t.activePlugins_[e] = !1,
                this.player = this.state = null,
                t[e] = Sn(e, _n[e])
            }
            static isBasic(e) {
                const t = "string" === typeof e ? vn(e) : e;
                return "function" === typeof t && !kn.prototype.isPrototypeOf(t.prototype)
            }
            static registerPlugin(e, t) {
                if ("string" !== typeof e)
                    throw new Error(`Illegal plugin name, "${e}", must be a string, was ${typeof e}.`);
                if (yn(e))
                    U.warn(`A plugin named "${e}" already exists. You may want to avoid re-registering plugins!`);
                else if (mn.prototype.hasOwnProperty(e))
                    throw new Error(`Illegal plugin name, "${e}", cannot share a name with an existing player method!`);
                if ("function" !== typeof t)
                    throw new Error(`Illegal plugin for "${e}", must be a function, was ${typeof t}.`);
                return _n[e] = t,
                e !== fn && (kn.isBasic(t) ? mn.prototype[e] = function(e, t) {
                    const i = function() {
                        bn(this, {
                            name: e,
                            plugin: t,
                            instance: null
                        }, !0);
                        const i = t.apply(this, arguments);
                        return Tn(this, e),
                        bn(this, {
                            name: e,
                            plugin: t,
                            instance: i
                        }),
                        i
                    };
                    return Object.keys(t).forEach((function(e) {
                        i[e] = t[e]
                    }
                    )),
                    i
                }(e, t) : mn.prototype[e] = Sn(e, t)),
                t
            }
            static deregisterPlugin(e) {
                if (e === fn)
                    throw new Error("Cannot de-register base plugin.");
                yn(e) && (delete _n[e],
                delete mn.prototype[e])
            }
            static getPlugins(e=Object.keys(_n)) {
                let t;
                return e.forEach((e=>{
                    const i = vn(e);
                    i && (t = t || {},
                    t[e] = i)
                }
                )),
                t
            }
            static getPluginVersion(e) {
                const t = vn(e);
                return t && t.VERSION || ""
            }
        }
        function Cn(e, t, i, s) {
            return function(e, t) {
                let i = !1;
                return function(...s) {
                    return i || U.warn(e),
                    i = !0,
                    t.apply(this, s)
                }
            }(`${t} is deprecated and will be removed in ${e}.0; please use ${i} instead.`, s)
        }
        kn.getPlugin = vn,
        kn.BASE_PLUGIN_NAME = fn,
        kn.registerPlugin(fn, kn),
        mn.prototype.usingPlugin = function(e) {
            return !!this.activePlugins_ && !0 === this.activePlugins_[e]
        }
        ,
        mn.prototype.hasPlugin = function(e) {
            return !!yn(e)
        }
        ;
        const En = e=>0 === e.indexOf("#") ? e.slice(1) : e;
        function wn(e, t, i) {
            let s = wn.getPlayer(e);
            if (s)
                return t && U.warn(`Player "${e}" is already initialised. Options will not be applied.`),
                i && s.ready(i),
                s;
            const r = "string" === typeof e ? qe("#" + En(e)) : e;
            if (!fe(r))
                throw new TypeError("The element or ID supplied is not valid. (videojs)");
            const a = "getRootNode"in r && r.getRootNode()instanceof n().ShadowRoot ? r.getRootNode() : r.ownerDocument.body;
            r.ownerDocument.defaultView && a.contains(r) || U.warn("The element supplied is not included in the DOM"),
            !0 === (t = t || {}).restoreEl && (t.restoreEl = (r.parentNode && r.parentNode.hasAttribute("data-vjs-player") ? r.parentNode : r).cloneNode(!0)),
            P("beforesetup").forEach((e=>{
                const i = e(r, V(t));
                q(i) && !Array.isArray(i) ? t = V(t, i) : U.error("please return an object in beforesetup hooks")
            }
            ));
            const o = Bt.getComponent("Player");
            return s = new o(r,t,i),
            P("setup").forEach((e=>e(s))),
            s
        }
        if (wn.hooks_ = I,
        wn.hooks = P,
        wn.hook = function(e, t) {
            P(e, t)
        }
        ,
        wn.hookOnce = function(e, t) {
            P(e, [].concat(t).map((t=>{
                const i = (...s)=>(A(e, i),
                t(...s));
                return i
            }
            )))
        }
        ,
        wn.removeHook = A,
        !0 !== n().VIDEOJS_NO_DYNAMIC_STYLE && ge()) {
            let e = qe(".vjs-styles-defaults");
            if (!e) {
                e = Je("vjs-styles-defaults");
                const t = qe("head");
                t && t.insertBefore(e, t.firstChild),
                Ze(e, "\n      .video-js {\n        width: 300px;\n        height: 150px;\n      }\n\n      .vjs-fluid:not(.vjs-audio-only-mode) {\n        padding-top: 56.25%\n      }\n    ")
            }
        }
        Xe(1, wn),
        wn.VERSION = x,
        wn.options = mn.prototype.options_,
        wn.getPlayers = ()=>mn.players,
        wn.getPlayer = e=>{
            const t = mn.players;
            let i;
            if ("string" === typeof e) {
                const s = En(e)
                  , n = t[s];
                if (n)
                    return n;
                i = qe("#" + s)
            } else
                i = e;
            if (fe(i)) {
                const {player: e, playerId: s} = i;
                if (e || t[s])
                    return e || t[s]
            }
        }
        ,
        wn.getAllPlayers = ()=>Object.keys(mn.players).map((e=>mn.players[e])).filter(Boolean),
        wn.players = mn.players,
        wn.getComponent = Bt.getComponent,
        wn.registerComponent = (e,t)=>(Ei.isTech(t) && U.warn(`The ${e} tech was registered as a component. It should instead be registered using videojs.registerTech(name, tech)`),
        Bt.registerComponent.call(Bt, e, t)),
        wn.getTech = Ei.getTech,
        wn.registerTech = Ei.registerTech,
        wn.use = function(e, t) {
            wi[e] = wi[e] || [],
            wi[e].push(t)
        }
        ,
        Object.defineProperty(wn, "middleware", {
            value: {},
            writeable: !1,
            enumerable: !0
        }),
        Object.defineProperty(wn.middleware, "TERMINATOR", {
            value: Ii,
            writeable: !1,
            enumerable: !0
        }),
        wn.browser = pe,
        wn.obj = G,
        wn.mergeOptions = Cn(9, "videojs.mergeOptions", "videojs.obj.merge", V),
        wn.defineLazyProperty = Cn(9, "videojs.defineLazyProperty", "videojs.obj.defineLazyProperty", W),
        wn.bind = Cn(9, "videojs.bind", "native Function.prototype.bind", gt),
        wn.registerPlugin = kn.registerPlugin,
        wn.deregisterPlugin = kn.deregisterPlugin,
        wn.plugin = (e,t)=>(U.warn("videojs.plugin() is deprecated; use videojs.registerPlugin() instead"),
        kn.registerPlugin(e, t)),
        wn.getPlugins = kn.getPlugins,
        wn.getPlugin = kn.getPlugin,
        wn.getPluginVersion = kn.getPluginVersion,
        wn.addLanguage = function(e, t) {
            return e = ("" + e).toLowerCase(),
            wn.options.languages = V(wn.options.languages, {
                [e]: t
            }),
            wn.options.languages[e]
        }
        ,
        wn.log = U,
        wn.createLogger = B,
        wn.time = Wt,
        wn.createTimeRange = Cn(9, "videojs.createTimeRange", "videojs.time.createTimeRanges", jt),
        wn.createTimeRanges = Cn(9, "videojs.createTimeRanges", "videojs.time.createTimeRanges", jt),
        wn.formatTime = Cn(9, "videojs.formatTime", "videojs.time.formatTime", zt),
        wn.setFormatTime = Cn(9, "videojs.setFormatTime", "videojs.time.setFormatTime", Ht),
        wn.resetFormatTime = Cn(9, "videojs.resetFormatTime", "videojs.time.resetFormatTime", Vt),
        wn.parseUrl = Cn(9, "videojs.parseUrl", "videojs.url.parseUrl", ui),
        wn.isCrossOrigin = Cn(9, "videojs.isCrossOrigin", "videojs.url.isCrossOrigin", mi),
        wn.EventTarget = Tt,
        wn.any = ct,
        wn.on = lt,
        wn.one = ut,
        wn.off = ht,
        wn.trigger = dt,
        wn.xhr = c(),
        wn.TextTrack = yi,
        wn.AudioTrack = vi,
        wn.VideoTrack = Ti,
        ["isEl", "isTextNode", "createEl", "hasClass", "addClass", "removeClass", "toggleClass", "setAttributes", "getAttributes", "emptyEl", "appendContent", "insertContent"].forEach((e=>{
            wn[e] = function() {
                return U.warn(`videojs.${e}() is deprecated; use videojs.dom.${e}() instead`),
                We[e].apply(null, arguments)
            }
        }
        )),
        wn.computedStyle = Cn(9, "videojs.computedStyle", "videojs.dom.computedStyle", Ve),
        wn.dom = We,
        wn.fn = yt,
        wn.num = ts,
        wn.str = Ut,
        wn.url = gi;
        class xn {
            constructor(e) {
                let t = this;
                return t.id = e.id,
                t.label = t.id,
                t.width = e.width,
                t.height = e.height,
                t.bitrate = e.bandwidth,
                t.frameRate = e.frameRate,
                t.enabled_ = e.enabled,
                Object.defineProperty(t, "enabled", {
                    get: ()=>t.enabled_(),
                    set(e) {
                        t.enabled_(e)
                    }
                }),
                t
            }
        }
        class In extends wn.EventTarget {
            constructor() {
                super();
                let e = this;
                return e.levels_ = [],
                e.selectedIndex_ = -1,
                Object.defineProperty(e, "selectedIndex", {
                    get: ()=>e.selectedIndex_
                }),
                Object.defineProperty(e, "length", {
                    get: ()=>e.levels_.length
                }),
                e[Symbol.iterator] = ()=>e.levels_.values(),
                e
            }
            addQualityLevel(e) {
                let t = this.getQualityLevelById(e.id);
                if (t)
                    return t;
                const i = this.levels_.length;
                return t = new xn(e),
                "" + i in this || Object.defineProperty(this, i, {
                    get() {
                        return this.levels_[i]
                    }
                }),
                this.levels_.push(t),
                this.trigger({
                    qualityLevel: t,
                    type: "addqualitylevel"
                }),
                t
            }
            removeQualityLevel(e) {
                let t = null;
                for (let i = 0, s = this.length; i < s; i++)
                    if (this[i] === e) {
                        t = this.levels_.splice(i, 1)[0],
                        this.selectedIndex_ === i ? this.selectedIndex_ = -1 : this.selectedIndex_ > i && this.selectedIndex_--;
                        break
                    }
                return t && this.trigger({
                    qualityLevel: e,
                    type: "removequalitylevel"
                }),
                t
            }
            getQualityLevelById(e) {
                for (let t = 0, i = this.length; t < i; t++) {
                    const i = this[t];
                    if (i.id === e)
                        return i
                }
                return null
            }
            dispose() {
                this.selectedIndex_ = -1,
                this.levels_.length = 0
            }
        }
        In.prototype.allowedEvents_ = {
            change: "change",
            addqualitylevel: "addqualitylevel",
            removequalitylevel: "removequalitylevel"
        };
        for (const tl in In.prototype.allowedEvents_)
            In.prototype["on" + tl] = null;
        var Pn = "4.0.0";
        const An = function(e) {
            return function(e, t) {
                const i = e.qualityLevels
                  , s = new In
                  , n = function() {
                    s.dispose(),
                    e.qualityLevels = i,
                    e.off("dispose", n)
                };
                return e.on("dispose", n),
                e.qualityLevels = ()=>s,
                e.qualityLevels.VERSION = Pn,
                s
            }(this, wn.obj.merge({}, e))
        };
        wn.registerPlugin("qualityLevels", An),
        An.VERSION = Pn;
        const Ln = f.Z
          , Dn = (e,t)=>t && t.responseURL && e !== t.responseURL ? t.responseURL : e
          , On = e=>wn.log.debug ? wn.log.debug.bind(wn, "VHS:", `${e} >`) : function() {}
        ;
        function Mn(...e) {
            const t = wn.obj || wn;
            return (t.merge || t.mergeOptions).apply(t, e)
        }
        function Rn(...e) {
            const t = wn.time || wn;
            return (t.createTimeRanges || t.createTimeRanges).apply(t, e)
        }
        const Un = 1 / 30
          , Bn = .1
          , Nn = function(e, t) {
            const i = [];
            let s;
            if (e && e.length)
                for (s = 0; s < e.length; s++)
                    t(e.start(s), e.end(s)) && i.push([e.start(s), e.end(s)]);
            return Rn(i)
        }
          , Fn = function(e, t) {
            return Nn(e, (function(e, i) {
                return e - Bn <= t && i + Bn >= t
            }
            ))
        }
          , jn = function(e, t) {
            return Nn(e, (function(e) {
                return e - Un >= t
            }
            ))
        }
          , $n = e=>{
            const t = [];
            if (!e || !e.length)
                return "";
            for (let i = 0; i < e.length; i++)
                t.push(e.start(i) + " => " + e.end(i));
            return t.join(", ")
        }
          , qn = e=>{
            const t = [];
            for (let i = 0; i < e.length; i++)
                t.push({
                    start: e.start(i),
                    end: e.end(i)
                });
            return t
        }
          , Hn = function(e) {
            if (e && e.length && e.end)
                return e.end(e.length - 1)
        }
          , Vn = function(e, t) {
            let i = 0;
            if (!e || !e.length)
                return i;
            for (let s = 0; s < e.length; s++) {
                const n = e.start(s)
                  , r = e.end(s);
                t > r || (i += t > n && t <= r ? r - t : r - n)
            }
            return i
        }
          , zn = (e,t)=>{
            if (!t.preload)
                return t.duration;
            let i = 0;
            return (t.parts || []).forEach((function(e) {
                i += e.duration
            }
            )),
            (t.preloadHints || []).forEach((function(t) {
                "PART" === t.type && (i += e.partTargetDuration)
            }
            )),
            i
        }
          , Wn = e=>(e.segments || []).reduce(((e,t,i)=>(t.parts ? t.parts.forEach((function(s, n) {
            e.push({
                duration: s.duration,
                segmentIndex: i,
                partIndex: n,
                part: s,
                segment: t
            })
        }
        )) : e.push({
            duration: t.duration,
            segmentIndex: i,
            partIndex: null,
            segment: t,
            part: null
        }),
        e)), [])
          , Gn = e=>{
            const t = e.segments && e.segments.length && e.segments[e.segments.length - 1];
            return t && t.parts || []
        }
          , Kn = ({preloadSegment: e})=>{
            if (!e)
                return;
            const {parts: t, preloadHints: i} = e;
            let s = (i || []).reduce(((e,t)=>e + ("PART" === t.type ? 1 : 0)), 0);
            return s += t && t.length ? t.length : 0,
            s
        }
          , Qn = (e,t)=>{
            if (t.endList)
                return 0;
            if (e && e.suggestedPresentationDelay)
                return e.suggestedPresentationDelay;
            const i = Gn(t).length > 0;
            return i && t.serverControl && t.serverControl.partHoldBack ? t.serverControl.partHoldBack : i && t.partTargetDuration ? 3 * t.partTargetDuration : t.serverControl && t.serverControl.holdBack ? t.serverControl.holdBack : t.targetDuration ? 3 * t.targetDuration : 0
        }
          , Xn = function(e, t, i) {
            if ("undefined" === typeof t && (t = e.mediaSequence + e.segments.length),
            t < e.mediaSequence)
                return 0;
            const s = function(e, t) {
                let i = 0
                  , s = t - e.mediaSequence
                  , n = e.segments[s];
                if (n) {
                    if ("undefined" !== typeof n.start)
                        return {
                            result: n.start,
                            precise: !0
                        };
                    if ("undefined" !== typeof n.end)
                        return {
                            result: n.end - n.duration,
                            precise: !0
                        }
                }
                for (; s--; ) {
                    if (n = e.segments[s],
                    "undefined" !== typeof n.end)
                        return {
                            result: i + n.end,
                            precise: !0
                        };
                    if (i += zn(e, n),
                    "undefined" !== typeof n.start)
                        return {
                            result: i + n.start,
                            precise: !0
                        }
                }
                return {
                    result: i,
                    precise: !1
                }
            }(e, t);
            if (s.precise)
                return s.result;
            const n = function(e, t) {
                let i, s = 0, n = t - e.mediaSequence;
                for (; n < e.segments.length; n++) {
                    if (i = e.segments[n],
                    "undefined" !== typeof i.start)
                        return {
                            result: i.start - s,
                            precise: !0
                        };
                    if (s += zn(e, i),
                    "undefined" !== typeof i.end)
                        return {
                            result: i.end - s,
                            precise: !0
                        }
                }
                return {
                    result: -1,
                    precise: !1
                }
            }(e, t);
            return n.precise ? n.result : s.result + i
        }
          , Yn = function(e, t, i) {
            if (!e)
                return 0;
            if ("number" !== typeof i && (i = 0),
            "undefined" === typeof t) {
                if (e.totalDuration)
                    return e.totalDuration;
                if (!e.endList)
                    return n()[1 / 0]
            }
            return Xn(e, t, i)
        }
          , Jn = function({defaultDuration: e, durationList: t, startIndex: i, endIndex: s}) {
            let n = 0;
            if (i > s && ([i,s] = [s, i]),
            i < 0) {
                for (let t = i; t < Math.min(0, s); t++)
                    n += e;
                i = 0
            }
            for (let r = i; r < s; r++)
                n += t[r].duration;
            return n
        }
          , Zn = function(e, t, i, s) {
            if (!e || !e.segments)
                return null;
            if (e.endList)
                return Yn(e);
            if (null === t)
                return null;
            t = t || 0;
            let n = Xn(e, e.mediaSequence + e.segments.length, t);
            return i && (n -= s = "number" === typeof s ? s : Qn(null, e)),
            Math.max(0, n)
        }
          , er = function(e) {
            return e.excludeUntil && e.excludeUntil > Date.now()
        }
          , tr = function(e) {
            return e.excludeUntil && e.excludeUntil === 1 / 0
        }
          , ir = function(e) {
            const t = er(e);
            return !e.disabled && !t
        }
          , sr = function(e, t) {
            return t.attributes && t.attributes[e]
        }
          , nr = (e,t)=>{
            if (1 === e.playlists.length)
                return !0;
            const i = t.attributes.BANDWIDTH || Number.MAX_VALUE;
            return 0 === e.playlists.filter((e=>!!ir(e) && (e.attributes.BANDWIDTH || 0) < i)).length
        }
          , rr = (e,t)=>!(!e && !t || !e && t || e && !t) && (e === t || (!(!e.id || !t.id || e.id !== t.id) || (!(!e.resolvedUri || !t.resolvedUri || e.resolvedUri !== t.resolvedUri) || !(!e.uri || !t.uri || e.uri !== t.uri))))
          , ar = function(e, t) {
            const i = e && e.mediaGroups && e.mediaGroups.AUDIO || {};
            let s = !1;
            for (const n in i) {
                for (const e in i[n])
                    if (s = t(i[n][e]),
                    s)
                        break;
                if (s)
                    break
            }
            return !!s
        }
          , or = e=>{
            if (!e || !e.playlists || !e.playlists.length) {
                return ar(e, (e=>e.playlists && e.playlists.length || e.uri))
            }
            for (let t = 0; t < e.playlists.length; t++) {
                const i = e.playlists[t]
                  , s = i.attributes && i.attributes.CODECS;
                if (s && s.split(",").every((e=>(0,
                y.KL)(e))))
                    continue;
                if (!ar(e, (e=>rr(i, e))))
                    return !1
            }
            return !0
        }
        ;
        var lr = {
            liveEdgeDelay: Qn,
            duration: Yn,
            seekable: function(e, t, i) {
                const s = t || 0;
                let n = Zn(e, t, !0, i);
                return null === n ? Rn() : (n < s && (n = s),
                Rn(s, n))
            },
            getMediaInfoForTime: function({playlist: e, currentTime: t, startingSegmentIndex: i, startingPartIndex: s, startTime: n, exactManifestTimings: r}) {
                let a = t - n;
                const o = Wn(e);
                let l = 0;
                for (let h = 0; h < o.length; h++) {
                    const e = o[h];
                    if (i === e.segmentIndex && ("number" !== typeof s || "number" !== typeof e.partIndex || s === e.partIndex)) {
                        l = h;
                        break
                    }
                }
                if (a < 0) {
                    if (l > 0)
                        for (let t = l - 1; t >= 0; t--) {
                            const i = o[t];
                            if (a += i.duration,
                            r) {
                                if (a < 0)
                                    continue
                            } else if (a + Un <= 0)
                                continue;
                            return {
                                partIndex: i.partIndex,
                                segmentIndex: i.segmentIndex,
                                startTime: n - Jn({
                                    defaultDuration: e.targetDuration,
                                    durationList: o,
                                    startIndex: l,
                                    endIndex: t
                                })
                            }
                        }
                    return {
                        partIndex: o[0] && o[0].partIndex || null,
                        segmentIndex: o[0] && o[0].segmentIndex || 0,
                        startTime: t
                    }
                }
                if (l < 0) {
                    for (let i = l; i < 0; i++)
                        if (a -= e.targetDuration,
                        a < 0)
                            return {
                                partIndex: o[0] && o[0].partIndex || null,
                                segmentIndex: o[0] && o[0].segmentIndex || 0,
                                startTime: t
                            };
                    l = 0
                }
                for (let h = l; h < o.length; h++) {
                    const t = o[h];
                    a -= t.duration;
                    const i = t.duration > Un && a + Un >= 0;
                    if (!(0 === a) && !i || h === o.length - 1) {
                        if (r) {
                            if (a > 0)
                                continue
                        } else if (a - Un >= 0)
                            continue;
                        return {
                            partIndex: t.partIndex,
                            segmentIndex: t.segmentIndex,
                            startTime: n + Jn({
                                defaultDuration: e.targetDuration,
                                durationList: o,
                                startIndex: l,
                                endIndex: h
                            })
                        }
                    }
                }
                return {
                    segmentIndex: o[o.length - 1].segmentIndex,
                    partIndex: o[o.length - 1].partIndex,
                    startTime: t
                }
            },
            isEnabled: ir,
            isDisabled: function(e) {
                return e.disabled
            },
            isExcluded: er,
            isIncompatible: tr,
            playlistEnd: Zn,
            isAes: function(e) {
                for (let t = 0; t < e.segments.length; t++)
                    if (e.segments[t].key)
                        return !0;
                return !1
            },
            hasAttribute: sr,
            estimateSegmentRequestTime: function(e, t, i, s=0) {
                if (!sr("BANDWIDTH", i))
                    return NaN;
                return (e * i.attributes.BANDWIDTH - 8 * s) / t
            },
            isLowestEnabledRendition: nr,
            isAudioOnly: or,
            playlistMatch: rr,
            segmentDurationWithParts: zn
        };
        const {log: hr} = wn
          , dr = (e,t)=>`${e}-${t}`
          , ur = (e,t,i)=>`placeholder-uri-${e}-${t}-${i}`
          , cr = (e,t)=>{
            e.mediaGroups && ["AUDIO", "SUBTITLES"].forEach((i=>{
                if (e.mediaGroups[i])
                    for (const s in e.mediaGroups[i])
                        for (const n in e.mediaGroups[i][s]) {
                            const r = e.mediaGroups[i][s][n];
                            t(r, i, s, n)
                        }
            }
            ))
        }
          , pr = ({playlist: e, uri: t, id: i})=>{
            e.id = i,
            e.playlistErrors_ = 0,
            t && (e.uri = t),
            e.attributes = e.attributes || {}
        }
          , mr = (e,t,i=ur)=>{
            e.uri = t;
            for (let n = 0; n < e.playlists.length; n++)
                if (!e.playlists[n].uri) {
                    const t = `placeholder-uri-${n}`;
                    e.playlists[n].uri = t
                }
            const s = or(e);
            cr(e, ((t,n,r,a)=>{
                if (!t.playlists || !t.playlists.length) {
                    if (s && "AUDIO" === n && !t.uri)
                        for (let t = 0; t < e.playlists.length; t++) {
                            const i = e.playlists[t];
                            if (i.attributes && i.attributes.AUDIO && i.attributes.AUDIO === r)
                                return
                        }
                    t.playlists = [(0,
                    g.Z)({}, t)]
                }
                t.playlists.forEach((function(t, s) {
                    const o = i(n, r, a, t)
                      , l = dr(s, o);
                    t.uri ? t.resolvedUri = t.resolvedUri || Ln(e.uri, t.uri) : (t.uri = 0 === s ? o : l,
                    t.resolvedUri = t.uri),
                    t.id = t.id || l,
                    t.attributes = t.attributes || {},
                    e.playlists[t.id] = t,
                    e.playlists[t.uri] = t
                }
                ))
            }
            )),
            (e=>{
                let t = e.playlists.length;
                for (; t--; ) {
                    const i = e.playlists[t];
                    pr({
                        playlist: i,
                        id: dr(t, i.uri)
                    }),
                    i.resolvedUri = Ln(e.uri, i.uri),
                    e.playlists[i.id] = i,
                    e.playlists[i.uri] = i,
                    i.attributes.BANDWIDTH || hr.warn("Invalid playlist STREAM-INF detected. Missing BANDWIDTH attribute.")
                }
            }
            )(e),
            (e=>{
                cr(e, (t=>{
                    t.uri && (t.resolvedUri = Ln(e.uri, t.uri))
                }
                ))
            }
            )(e)
        }
        ;
        class gr {
            constructor() {
                this.offset_ = null,
                this.pendingDateRanges_ = new Map,
                this.processedDateRanges_ = new Map
            }
            setOffset(e=[]) {
                if (null !== this.offset_)
                    return;
                if (!e.length)
                    return;
                const [t] = e;
                void 0 !== t.programDateTime && (this.offset_ = t.programDateTime / 1e3)
            }
            setPendingDateRanges(e=[]) {
                if (!e.length)
                    return;
                const [t] = e
                  , i = t.startDate.getTime();
                this.trimProcessedDateRanges_(i),
                this.pendingDateRanges_ = e.reduce(((e,t)=>(e.set(t.id, t),
                e)), new Map)
            }
            processDateRange(e) {
                this.pendingDateRanges_.delete(e.id),
                this.processedDateRanges_.set(e.id, e)
            }
            getDateRangesToProcess() {
                if (null === this.offset_)
                    return [];
                const e = {}
                  , t = [];
                this.pendingDateRanges_.forEach(((i,s)=>{
                    if (!this.processedDateRanges_.has(s) && (i.startTime = i.startDate.getTime() / 1e3 - this.offset_,
                    i.processDateRange = ()=>this.processDateRange(i),
                    t.push(i),
                    i.class))
                        if (e[i.class]) {
                            const t = e[i.class].push(i);
                            i.classListIndex = t - 1
                        } else
                            e[i.class] = [i],
                            i.classListIndex = 0
                }
                ));
                for (const i of t) {
                    const t = e[i.class] || [];
                    i.endDate ? i.endTime = i.endDate.getTime() / 1e3 - this.offset_ : i.endOnNext && t[i.classListIndex + 1] ? i.endTime = t[i.classListIndex + 1].startTime : i.duration ? i.endTime = i.startTime + i.duration : i.plannedDuration ? i.endTime = i.startTime + i.plannedDuration : i.endTime = i.startTime
                }
                return t
            }
            trimProcessedDateRanges_(e) {
                new Map(this.processedDateRanges_).forEach(((t,i)=>{
                    t.startDate.getTime() < e && this.processedDateRanges_.delete(i)
                }
                ))
            }
        }
        const {EventTarget: fr} = wn
          , _r = (e,t)=>{
            if (!e)
                return t;
            const i = Mn(e, t);
            if (e.preloadHints && !t.preloadHints && delete i.preloadHints,
            e.parts && !t.parts)
                delete i.parts;
            else if (e.parts && t.parts)
                for (let s = 0; s < t.parts.length; s++)
                    e.parts && e.parts[s] && (i.parts[s] = Mn(e.parts[s], t.parts[s]));
            return !e.skipped && t.skipped && (i.skipped = !1),
            e.preload && !t.preload && (i.preload = !1),
            i
        }
          , yr = (e,t)=>{
            !e.resolvedUri && e.uri && (e.resolvedUri = Ln(t, e.uri)),
            e.key && !e.key.resolvedUri && (e.key.resolvedUri = Ln(t, e.key.uri)),
            e.map && !e.map.resolvedUri && (e.map.resolvedUri = Ln(t, e.map.uri)),
            e.map && e.map.key && !e.map.key.resolvedUri && (e.map.key.resolvedUri = Ln(t, e.map.key.uri)),
            e.parts && e.parts.length && e.parts.forEach((e=>{
                e.resolvedUri || (e.resolvedUri = Ln(t, e.uri))
            }
            )),
            e.preloadHints && e.preloadHints.length && e.preloadHints.forEach((e=>{
                e.resolvedUri || (e.resolvedUri = Ln(t, e.uri))
            }
            ))
        }
          , vr = function(e) {
            const t = e.segments || []
              , i = e.preloadSegment;
            if (i && i.parts && i.parts.length) {
                if (i.preloadHints)
                    for (let e = 0; e < i.preloadHints.length; e++)
                        if ("MAP" === i.preloadHints[e].type)
                            return t;
                i.duration = e.targetDuration,
                i.preload = !0,
                t.push(i)
            }
            return t
        }
          , Tr = (e,t)=>e === t || e.segments && t.segments && e.segments.length === t.segments.length && e.endList === t.endList && e.mediaSequence === t.mediaSequence && e.preloadSegment === t.preloadSegment
          , br = (e,t,i=Tr)=>{
            const s = Mn(e, {})
              , n = s.playlists[t.id];
            if (!n)
                return null;
            if (i(n, t))
                return null;
            t.segments = vr(t);
            const r = Mn(n, t);
            if (r.preloadSegment && !t.preloadSegment && delete r.preloadSegment,
            n.segments) {
                if (t.skip) {
                    t.segments = t.segments || [];
                    for (let e = 0; e < t.skip.skippedSegments; e++)
                        t.segments.unshift({
                            skipped: !0
                        })
                }
                r.segments = ((e,t,i)=>{
                    const s = e.slice()
                      , n = t.slice();
                    i = i || 0;
                    const r = [];
                    let a;
                    for (let o = 0; o < n.length; o++) {
                        const e = s[o + i]
                          , t = n[o];
                        e ? (a = e.map || a,
                        r.push(_r(e, t))) : (a && !t.map && (t.map = a),
                        r.push(t))
                    }
                    return r
                }
                )(n.segments, t.segments, t.mediaSequence - n.mediaSequence)
            }
            r.segments.forEach((e=>{
                yr(e, r.resolvedUri)
            }
            ));
            for (let a = 0; a < s.playlists.length; a++)
                s.playlists[a].id === t.id && (s.playlists[a] = r);
            return s.playlists[t.id] = r,
            s.playlists[t.uri] = r,
            cr(e, ((e,i,s,n)=>{
                if (e.playlists)
                    for (let a = 0; a < e.playlists.length; a++)
                        t.id === e.playlists[a].id && (e.playlists[a] = r)
            }
            )),
            s
        }
          , Sr = (e,t)=>{
            const i = e.segments || []
              , s = i[i.length - 1]
              , n = s && s.parts && s.parts[s.parts.length - 1]
              , r = n && n.duration || s && s.duration;
            return t && r ? 1e3 * r : 500 * (e.partTargetDuration || e.targetDuration || 10)
        }
        ;
        class kr extends fr {
            constructor(e, t, i={}) {
                if (super(),
                !e)
                    throw new Error("A non-empty playlist URL or object is required");
                this.logger_ = On("PlaylistLoader");
                const {withCredentials: s=!1} = i;
                this.src = e,
                this.vhs_ = t,
                this.withCredentials = s,
                this.addDateRangesToTextTrack_ = i.addDateRangesToTextTrack;
                const n = t.options_;
                this.customTagParsers = n && n.customTagParsers || [],
                this.customTagMappers = n && n.customTagMappers || [],
                this.llhls = n && n.llhls,
                this.dateRangesStorage_ = new gr,
                this.state = "HAVE_NOTHING",
                this.handleMediaupdatetimeout_ = this.handleMediaupdatetimeout_.bind(this),
                this.on("mediaupdatetimeout", this.handleMediaupdatetimeout_),
                this.on("loadedplaylist", this.handleLoadedPlaylist_.bind(this))
            }
            handleLoadedPlaylist_() {
                const e = this.media();
                if (!e)
                    return;
                this.dateRangesStorage_.setOffset(e.segments),
                this.dateRangesStorage_.setPendingDateRanges(e.dateRanges);
                const t = this.dateRangesStorage_.getDateRangesToProcess();
                t.length && this.addDateRangesToTextTrack_ && this.addDateRangesToTextTrack_(t)
            }
            handleMediaupdatetimeout_() {
                if ("HAVE_METADATA" !== this.state)
                    return;
                const e = this.media();
                let t = Ln(this.main.uri, e.uri);
                this.llhls && (t = ((e,t)=>{
                    if (t.endList || !t.serverControl)
                        return e;
                    const i = {};
                    if (t.serverControl.canBlockReload) {
                        const {preloadSegment: e} = t;
                        let s = t.mediaSequence + t.segments.length;
                        if (e) {
                            const n = e.parts || []
                              , r = Kn(t) - 1;
                            r > -1 && r !== n.length - 1 && (i._HLS_part = r),
                            (r > -1 || n.length) && s--
                        }
                        i._HLS_msn = s
                    }
                    if (t.serverControl && t.serverControl.canSkipUntil && (i._HLS_skip = t.serverControl.canSkipDateranges ? "v2" : "YES"),
                    Object.keys(i).length) {
                        const t = new (n().URL)(e);
                        ["_HLS_skip", "_HLS_msn", "_HLS_part"].forEach((function(e) {
                            i.hasOwnProperty(e) && t.searchParams.set(e, i[e])
                        }
                        )),
                        e = t.toString()
                    }
                    return e
                }
                )(t, e)),
                this.state = "HAVE_CURRENT_METADATA",
                this.request = this.vhs_.xhr({
                    uri: t,
                    withCredentials: this.withCredentials
                }, ((e,t)=>{
                    if (this.request)
                        return e ? this.playlistRequestError(this.request, this.media(), "HAVE_METADATA") : void this.haveMetadata({
                            playlistString: this.request.responseText,
                            url: this.media().uri,
                            id: this.media().id
                        })
                }
                ))
            }
            playlistRequestError(e, t, i) {
                const {uri: s, id: n} = t;
                this.request = null,
                i && (this.state = i),
                this.error = {
                    playlist: this.main.playlists[n],
                    status: e.status,
                    message: `HLS playlist request error at URL: ${s}.`,
                    responseText: e.responseText,
                    code: e.status >= 500 ? 4 : 2
                },
                this.trigger("error")
            }
            parseManifest_({url: e, manifestString: t}) {
                return (({onwarn: e, oninfo: t, manifestString: i, customTagParsers: s=[], customTagMappers: n=[], llhls: r})=>{
                    const a = new _._b;
                    e && a.on("warn", e),
                    t && a.on("info", t),
                    s.forEach((e=>a.addParser(e))),
                    n.forEach((e=>a.addTagMapper(e))),
                    a.push(i),
                    a.end();
                    const o = a.manifest;
                    if (r || (["preloadSegment", "skip", "serverControl", "renditionReports", "partInf", "partTargetDuration"].forEach((function(e) {
                        o.hasOwnProperty(e) && delete o[e]
                    }
                    )),
                    o.segments && o.segments.forEach((function(e) {
                        ["parts", "preloadHints"].forEach((function(t) {
                            e.hasOwnProperty(t) && delete e[t]
                        }
                        ))
                    }
                    ))),
                    !o.targetDuration) {
                        let t = 10;
                        o.segments && o.segments.length && (t = o.segments.reduce(((e,t)=>Math.max(e, t.duration)), 0)),
                        e && e({
                            message: `manifest has no targetDuration defaulting to ${t}`
                        }),
                        o.targetDuration = t
                    }
                    const l = Gn(o);
                    if (l.length && !o.partTargetDuration) {
                        const t = l.reduce(((e,t)=>Math.max(e, t.duration)), 0);
                        e && (e({
                            message: `manifest has no partTargetDuration defaulting to ${t}`
                        }),
                        hr.error("LL-HLS manifest has parts but lacks required #EXT-X-PART-INF:PART-TARGET value. See https://datatracker.ietf.org/doc/html/draft-pantos-hls-rfc8216bis-09#section-4.4.3.7. Playback is not guaranteed.")),
                        o.partTargetDuration = t
                    }
                    return o
                }
                )({
                    onwarn: ({message: t})=>this.logger_(`m3u8-parser warn for ${e}: ${t}`),
                    oninfo: ({message: t})=>this.logger_(`m3u8-parser info for ${e}: ${t}`),
                    manifestString: t,
                    customTagParsers: this.customTagParsers,
                    customTagMappers: this.customTagMappers,
                    llhls: this.llhls
                })
            }
            haveMetadata({playlistString: e, playlistObject: t, url: i, id: s}) {
                this.request = null,
                this.state = "HAVE_METADATA";
                const n = t || this.parseManifest_({
                    url: i,
                    manifestString: e
                });
                n.lastRequest = Date.now(),
                pr({
                    playlist: n,
                    uri: i,
                    id: s
                });
                const r = br(this.main, n);
                this.targetDuration = n.partTargetDuration || n.targetDuration,
                this.pendingMedia_ = null,
                r ? (this.main = r,
                this.media_ = this.main.playlists[s]) : this.trigger("playlistunchanged"),
                this.updateMediaUpdateTimeout_(Sr(this.media(), !!r)),
                this.trigger("loadedplaylist")
            }
            dispose() {
                this.trigger("dispose"),
                this.stopRequest(),
                n().clearTimeout(this.mediaUpdateTimeout),
                n().clearTimeout(this.finalRenditionTimeout),
                this.dateRangesStorage_ = new gr,
                this.off()
            }
            stopRequest() {
                if (this.request) {
                    const e = this.request;
                    this.request = null,
                    e.onreadystatechange = null,
                    e.abort()
                }
            }
            media(e, t) {
                if (!e)
                    return this.media_;
                if ("HAVE_NOTHING" === this.state)
                    throw new Error("Cannot switch media playlist from " + this.state);
                if ("string" === typeof e) {
                    if (!this.main.playlists[e])
                        throw new Error("Unknown playlist URI: " + e);
                    e = this.main.playlists[e]
                }
                if (n().clearTimeout(this.finalRenditionTimeout),
                t) {
                    const t = (e.partTargetDuration || e.targetDuration) / 2 * 1e3 || 5e3;
                    return void (this.finalRenditionTimeout = n().setTimeout(this.media.bind(this, e, !1), t))
                }
                const i = this.state
                  , s = !this.media_ || e.id !== this.media_.id
                  , r = this.main.playlists[e.id];
                if (r && r.endList || e.endList && e.segments.length)
                    return this.request && (this.request.onreadystatechange = null,
                    this.request.abort(),
                    this.request = null),
                    this.state = "HAVE_METADATA",
                    this.media_ = e,
                    void (s && (this.trigger("mediachanging"),
                    "HAVE_MAIN_MANIFEST" === i ? this.trigger("loadedmetadata") : this.trigger("mediachange")));
                if (this.updateMediaUpdateTimeout_(Sr(e, !0)),
                s) {
                    if (this.state = "SWITCHING_MEDIA",
                    this.request) {
                        if (e.resolvedUri === this.request.url)
                            return;
                        this.request.onreadystatechange = null,
                        this.request.abort(),
                        this.request = null
                    }
                    this.media_ && this.trigger("mediachanging"),
                    this.pendingMedia_ = e,
                    this.request = this.vhs_.xhr({
                        uri: e.resolvedUri,
                        withCredentials: this.withCredentials
                    }, ((t,s)=>{
                        if (this.request) {
                            if (e.lastRequest = Date.now(),
                            e.resolvedUri = Dn(e.resolvedUri, s),
                            t)
                                return this.playlistRequestError(this.request, e, i);
                            this.haveMetadata({
                                playlistString: s.responseText,
                                url: e.uri,
                                id: e.id
                            }),
                            "HAVE_MAIN_MANIFEST" === i ? this.trigger("loadedmetadata") : this.trigger("mediachange")
                        }
                    }
                    ))
                }
            }
            pause() {
                this.mediaUpdateTimeout && (n().clearTimeout(this.mediaUpdateTimeout),
                this.mediaUpdateTimeout = null),
                this.stopRequest(),
                "HAVE_NOTHING" === this.state && (this.started = !1),
                "SWITCHING_MEDIA" === this.state ? this.media_ ? this.state = "HAVE_METADATA" : this.state = "HAVE_MAIN_MANIFEST" : "HAVE_CURRENT_METADATA" === this.state && (this.state = "HAVE_METADATA")
            }
            load(e) {
                this.mediaUpdateTimeout && (n().clearTimeout(this.mediaUpdateTimeout),
                this.mediaUpdateTimeout = null);
                const t = this.media();
                if (e) {
                    const e = t ? (t.partTargetDuration || t.targetDuration) / 2 * 1e3 : 5e3;
                    this.mediaUpdateTimeout = n().setTimeout((()=>{
                        this.mediaUpdateTimeout = null,
                        this.load()
                    }
                    ), e)
                } else
                    this.started ? t && !t.endList ? this.trigger("mediaupdatetimeout") : this.trigger("loadedplaylist") : this.start()
            }
            updateMediaUpdateTimeout_(e) {
                this.mediaUpdateTimeout && (n().clearTimeout(this.mediaUpdateTimeout),
                this.mediaUpdateTimeout = null),
                this.media() && !this.media().endList && (this.mediaUpdateTimeout = n().setTimeout((()=>{
                    this.mediaUpdateTimeout = null,
                    this.trigger("mediaupdatetimeout"),
                    this.updateMediaUpdateTimeout_(e)
                }
                ), e))
            }
            start() {
                if (this.started = !0,
                "object" === typeof this.src)
                    return this.src.uri || (this.src.uri = n().location.href),
                    this.src.resolvedUri = this.src.uri,
                    void setTimeout((()=>{
                        this.setupInitialPlaylist(this.src)
                    }
                    ), 0);
                this.request = this.vhs_.xhr({
                    uri: this.src,
                    withCredentials: this.withCredentials
                }, ((e,t)=>{
                    if (!this.request)
                        return;
                    if (this.request = null,
                    e)
                        return this.error = {
                            status: t.status,
                            message: `HLS playlist request error at URL: ${this.src}.`,
                            responseText: t.responseText,
                            code: 2
                        },
                        "HAVE_NOTHING" === this.state && (this.started = !1),
                        this.trigger("error");
                    this.src = Dn(this.src, t);
                    const i = this.parseManifest_({
                        manifestString: t.responseText,
                        url: this.src
                    });
                    this.setupInitialPlaylist(i)
                }
                ))
            }
            srcUri() {
                return "string" === typeof this.src ? this.src : this.src.uri
            }
            setupInitialPlaylist(e) {
                if (this.state = "HAVE_MAIN_MANIFEST",
                e.playlists)
                    return this.main = e,
                    mr(this.main, this.srcUri()),
                    e.playlists.forEach((e=>{
                        e.segments = vr(e),
                        e.segments.forEach((t=>{
                            yr(t, e.resolvedUri)
                        }
                        ))
                    }
                    )),
                    this.trigger("loadedplaylist"),
                    void (this.request || this.media(this.main.playlists[0]));
                const t = this.srcUri() || n().location.href;
                this.main = ((e,t)=>{
                    const i = dr(0, t)
                      , s = {
                        mediaGroups: {
                            AUDIO: {},
                            VIDEO: {},
                            "CLOSED-CAPTIONS": {},
                            SUBTITLES: {}
                        },
                        uri: n().location.href,
                        resolvedUri: n().location.href,
                        playlists: [{
                            uri: t,
                            id: i,
                            resolvedUri: t,
                            attributes: {}
                        }]
                    };
                    return s.playlists[i] = s.playlists[0],
                    s.playlists[t] = s.playlists[0],
                    s
                }
                )(0, t),
                this.haveMetadata({
                    playlistObject: e,
                    url: t,
                    id: this.main.playlists[0].id
                }),
                this.trigger("loadedmetadata")
            }
            updateOrDeleteClone(e, t) {
                const i = this.main
                  , s = e.ID;
                let n = i.playlists.length;
                for (; n--; ) {
                    const r = i.playlists[n];
                    if (r.attributes["PATHWAY-ID"] === s) {
                        const a = r.resolvedUri
                          , o = r.id;
                        if (t) {
                            const t = this.createCloneURI_(r.resolvedUri, e)
                              , a = dr(s, t)
                              , o = this.createCloneAttributes_(s, r.attributes)
                              , l = this.createClonePlaylist_(r, a, e, o);
                            i.playlists[n] = l,
                            i.playlists[a] = l,
                            i.playlists[t] = l
                        } else
                            i.playlists.splice(n, 1);
                        delete i.playlists[o],
                        delete i.playlists[a]
                    }
                }
                this.updateOrDeleteCloneMedia(e, t)
            }
            updateOrDeleteCloneMedia(e, t) {
                const i = this.main
                  , s = e.ID;
                ["AUDIO", "SUBTITLES", "CLOSED-CAPTIONS"].forEach((e=>{
                    if (i.mediaGroups[e] && i.mediaGroups[e][s])
                        for (const t in i.mediaGroups[e])
                            if (t === s) {
                                for (const s in i.mediaGroups[e][t]) {
                                    i.mediaGroups[e][t][s].playlists.forEach(((e,t)=>{
                                        const s = i.playlists[e.id]
                                          , n = s.id
                                          , r = s.resolvedUri;
                                        delete i.playlists[n],
                                        delete i.playlists[r]
                                    }
                                    ))
                                }
                                delete i.mediaGroups[e][t]
                            }
                }
                )),
                t && this.createClonedMediaGroups_(e)
            }
            addClonePathway(e, t={}) {
                const i = this.main
                  , s = i.playlists.length
                  , n = this.createCloneURI_(t.resolvedUri, e)
                  , r = dr(e.ID, n)
                  , a = this.createCloneAttributes_(e.ID, t.attributes)
                  , o = this.createClonePlaylist_(t, r, e, a);
                i.playlists[s] = o,
                i.playlists[r] = o,
                i.playlists[n] = o,
                this.createClonedMediaGroups_(e)
            }
            createClonedMediaGroups_(e) {
                const t = e.ID
                  , i = e["BASE-ID"]
                  , s = this.main;
                ["AUDIO", "SUBTITLES", "CLOSED-CAPTIONS"].forEach((n=>{
                    if (s.mediaGroups[n] && !s.mediaGroups[n][t])
                        for (const r in s.mediaGroups[n])
                            if (r === i) {
                                s.mediaGroups[n][t] = {};
                                for (const i in s.mediaGroups[n][r]) {
                                    const a = s.mediaGroups[n][r][i];
                                    s.mediaGroups[n][t][i] = (0,
                                    g.Z)({}, a);
                                    const o = s.mediaGroups[n][t][i]
                                      , l = this.createCloneURI_(a.resolvedUri, e);
                                    o.resolvedUri = l,
                                    o.uri = l,
                                    o.playlists = [],
                                    a.playlists.forEach(((r,a)=>{
                                        const l = s.playlists[r.id]
                                          , h = ur(n, t, i)
                                          , d = dr(t, h);
                                        if (l && !s.playlists[d]) {
                                            const t = this.createClonePlaylist_(l, d, e)
                                              , i = t.resolvedUri;
                                            s.playlists[d] = t,
                                            s.playlists[i] = t
                                        }
                                        o.playlists[a] = this.createClonePlaylist_(r, d, e)
                                    }
                                    ))
                                }
                            }
                }
                ))
            }
            createClonePlaylist_(e, t, i, s) {
                const n = this.createCloneURI_(e.resolvedUri, i)
                  , r = {
                    resolvedUri: n,
                    uri: n,
                    id: t
                };
                return e.segments && (r.segments = []),
                s && (r.attributes = s),
                Mn(e, r)
            }
            createCloneURI_(e, t) {
                const i = new URL(e);
                i.hostname = t["URI-REPLACEMENT"].HOST;
                const s = t["URI-REPLACEMENT"].PARAMS;
                for (const n of Object.keys(s))
                    i.searchParams.set(n, s[n]);
                return i.href
            }
            createCloneAttributes_(e, t) {
                const i = {
                    "PATHWAY-ID": e
                };
                return ["AUDIO", "SUBTITLES", "CLOSED-CAPTIONS"].forEach((s=>{
                    t[s] && (i[s] = e)
                }
                )),
                i
            }
            getKeyIdSet(e) {
                if (e.contentProtection) {
                    const t = new Set;
                    for (const i in e.contentProtection) {
                        const s = e.contentProtection[i].attributes.keyId;
                        s && t.add(s.toLowerCase())
                    }
                    return t
                }
            }
        }
        const {xhr: Cr} = wn
          , Er = function(e, t, i, s) {
            const n = "arraybuffer" === e.responseType ? e.response : e.responseText;
            !t && n && (e.responseTime = Date.now(),
            e.roundTripTime = e.responseTime - e.requestTime,
            e.bytesReceived = n.byteLength || n.length,
            e.bandwidth || (e.bandwidth = Math.floor(e.bytesReceived / e.roundTripTime * 8 * 1e3))),
            i.headers && (e.responseHeaders = i.headers),
            t && "ETIMEDOUT" === t.code && (e.timedout = !0),
            t || e.aborted || 200 === i.statusCode || 206 === i.statusCode || 0 === i.statusCode || (t = new Error("XHR Failed with a response of: " + (e && (n || e.responseText)))),
            s(t, e)
        }
          , wr = function() {
            const e = function e(t, i) {
                t = Mn({
                    timeout: 45e3
                }, t);
                const s = e.beforeRequest || wn.Vhs.xhr.beforeRequest
                  , n = e._requestCallbackSet || wn.Vhs.xhr._requestCallbackSet || new Set
                  , r = e._responseCallbackSet || wn.Vhs.xhr._responseCallbackSet;
                s && "function" === typeof s && (wn.log.warn("beforeRequest is deprecated, use onRequest instead."),
                n.add(s));
                const a = !0 === wn.Vhs.xhr.original ? Cr : wn.Vhs.xhr
                  , o = ((e,t)=>{
                    if (!e || !e.size)
                        return;
                    let i = t;
                    return e.forEach((e=>{
                        i = e(i)
                    }
                    )),
                    i
                }
                )(n, t);
                n.delete(s);
                const l = a(o || t, (function(e, t) {
                    return ((e,t,i,s)=>{
                        e && e.size && e.forEach((e=>{
                            e(t, i, s)
                        }
                        ))
                    }
                    )(r, l, e, t),
                    Er(l, e, t, i)
                }
                ))
                  , h = l.abort;
                return l.abort = function() {
                    return l.aborted = !0,
                    h.apply(l, arguments)
                }
                ,
                l.uri = t.uri,
                l.requestTime = Date.now(),
                l
            };
            return e.original = !0,
            e
        }
          , xr = function(e) {
            const t = {};
            return e.byterange && (t.Range = function(e) {
                let t;
                const i = e.offset;
                return t = "bigint" === typeof e.offset || "bigint" === typeof e.length ? n().BigInt(e.offset) + n().BigInt(e.length) - n().BigInt(1) : e.offset + e.length - 1,
                "bytes=" + i + "-" + t
            }(e.byterange)),
            t
        }
          , Ir = function(e, t) {
            return e.start(t) + "-" + e.end(t)
        }
          , Pr = function(e, t) {
            const i = e.toString(16);
            return "00".substring(0, 2 - i.length) + i + (t % 2 ? " " : "")
        }
          , Ar = function(e) {
            return e >= 32 && e < 126 ? String.fromCharCode(e) : "."
        }
          , Lr = function(e) {
            const t = {};
            return Object.keys(e).forEach((i=>{
                const s = e[i];
                (0,
                T.Au)(s) ? t[i] = {
                    bytes: s.buffer,
                    byteOffset: s.byteOffset,
                    byteLength: s.byteLength
                } : t[i] = s
            }
            )),
            t
        }
          , Dr = function(e) {
            const t = e.byterange || {
                length: 1 / 0,
                offset: 0
            };
            return [t.length, t.offset, e.resolvedUri].join(",")
        }
          , Or = function(e) {
            return e.resolvedUri
        }
          , Mr = e=>{
            const t = Array.prototype.slice.call(e)
              , i = 16;
            let s, n, r = "";
            for (let a = 0; a < t.length / i; a++)
                s = t.slice(a * i, a * i + i).map(Pr).join(""),
                n = t.slice(a * i, a * i + i).map(Ar).join(""),
                r += s + " " + n + "\n";
            return r
        }
        ;
        var Rr = Object.freeze({
            __proto__: null,
            createTransferableMessage: Lr,
            initSegmentId: Dr,
            segmentKeyId: Or,
            hexDump: Mr,
            tagDump: ({bytes: e})=>Mr(e),
            textRanges: e=>{
                let t, i = "";
                for (t = 0; t < e.length; t++)
                    i += Ir(e, t) + " ";
                return i
            }
        });
        const Ur = ({playlist: e, time: t, callback: i})=>{
            if (!i)
                throw new Error("getProgramTime: callback must be provided");
            if (!e || void 0 === t)
                return i({
                    message: "getProgramTime: playlist and time must be provided"
                });
            const s = ((e,t)=>{
                if (!t || !t.segments || 0 === t.segments.length)
                    return null;
                let i, s = 0;
                for (let r = 0; r < t.segments.length && (i = t.segments[r],
                s = i.videoTimingInfo ? i.videoTimingInfo.transmuxedPresentationEnd : s + i.duration,
                !(e <= s)); r++)
                    ;
                const n = t.segments[t.segments.length - 1];
                if (n.videoTimingInfo && n.videoTimingInfo.transmuxedPresentationEnd < e)
                    return null;
                if (e > s) {
                    if (e > s + .25 * n.duration)
                        return null;
                    i = n
                }
                return {
                    segment: i,
                    estimatedStart: i.videoTimingInfo ? i.videoTimingInfo.transmuxedPresentationStart : s - i.duration,
                    type: i.videoTimingInfo ? "accurate" : "estimate"
                }
            }
            )(t, e);
            if (!s)
                return i({
                    message: "valid programTime was not found"
                });
            if ("estimate" === s.type)
                return i({
                    message: "Accurate programTime could not be determined. Please seek to e.seekTime and try again",
                    seekTime: s.estimatedStart
                });
            const n = {
                mediaSeconds: t
            }
              , r = ((e,t)=>{
                if (!t.dateTimeObject)
                    return null;
                const i = t.videoTimingInfo.transmuxerPrependedSeconds
                  , s = e - (t.videoTimingInfo.transmuxedPresentationStart + i);
                return new Date(t.dateTimeObject.getTime() + 1e3 * s)
            }
            )(t, s.segment);
            return r && (n.programDateTime = r.toISOString()),
            i(null, n)
        }
          , Br = ({programTime: e, playlist: t, retryCount: i=2, seekTo: s, pauseAfterSeek: n=!0, tech: r, callback: a})=>{
            if (!a)
                throw new Error("seekToProgramTime: callback must be provided");
            if ("undefined" === typeof e || !t || !s)
                return a({
                    message: "seekToProgramTime: programTime, seekTo and playlist must be provided"
                });
            if (!t.endList && !r.hasStarted_)
                return a({
                    message: "player must be playing a live stream to start buffering"
                });
            if (!(e=>{
                if (!e.segments || 0 === e.segments.length)
                    return !1;
                for (let t = 0; t < e.segments.length; t++)
                    if (!e.segments[t].dateTimeObject)
                        return !1;
                return !0
            }
            )(t))
                return a({
                    message: "programDateTime tags must be provided in the manifest " + t.resolvedUri
                });
            const o = ((e,t)=>{
                let i;
                try {
                    i = new Date(e)
                } catch (l) {
                    return null
                }
                if (!t || !t.segments || 0 === t.segments.length)
                    return null;
                let s = t.segments[0];
                if (i < new Date(s.dateTimeObject))
                    return null;
                for (let h = 0; h < t.segments.length - 1 && (s = t.segments[h],
                !(i < new Date(t.segments[h + 1].dateTimeObject))); h++)
                    ;
                const n = t.segments[t.segments.length - 1]
                  , r = n.dateTimeObject
                  , a = n.videoTimingInfo ? (o = n.videoTimingInfo).transmuxedPresentationEnd - o.transmuxedPresentationStart - o.transmuxerPrependedSeconds : n.duration + .25 * n.duration;
                var o;
                return i > new Date(r.getTime() + 1e3 * a) ? null : (i > new Date(r) && (s = n),
                {
                    segment: s,
                    estimatedStart: s.videoTimingInfo ? s.videoTimingInfo.transmuxedPresentationStart : lr.duration(t, t.mediaSequence + t.segments.indexOf(s)),
                    type: s.videoTimingInfo ? "accurate" : "estimate"
                })
            }
            )(e, t);
            if (!o)
                return a({
                    message: `${e} was not found in the stream`
                });
            const l = o.segment
              , h = ((e,t)=>{
                let i, s;
                try {
                    i = new Date(e),
                    s = new Date(t)
                } catch (r) {}
                const n = i.getTime();
                return (s.getTime() - n) / 1e3
            }
            )(l.dateTimeObject, e);
            if ("estimate" === o.type)
                return 0 === i ? a({
                    message: `${e} is not buffered yet. Try again`
                }) : (s(o.estimatedStart + h),
                void r.one("seeked", (()=>{
                    Br({
                        programTime: e,
                        playlist: t,
                        retryCount: i - 1,
                        seekTo: s,
                        pauseAfterSeek: n,
                        tech: r,
                        callback: a
                    })
                }
                )));
            const d = l.start + h;
            r.one("seeked", (()=>a(null, r.currentTime()))),
            n && r.pause(),
            s(d)
        }
          , Nr = (e,t)=>{
            if (4 === e.readyState)
                return t()
        }
          , {EventTarget: Fr} = wn
          , jr = function(e, t) {
            if (!Tr(e, t))
                return !1;
            if (e.sidx && t.sidx && (e.sidx.offset !== t.sidx.offset || e.sidx.length !== t.sidx.length))
                return !1;
            if (!e.sidx && t.sidx || e.sidx && !t.sidx)
                return !1;
            if (e.segments && !t.segments || !e.segments && t.segments)
                return !1;
            if (!e.segments && !t.segments)
                return !0;
            for (let i = 0; i < e.segments.length; i++) {
                const s = e.segments[i]
                  , n = t.segments[i];
                if (s.uri !== n.uri)
                    return !1;
                if (!s.byterange && !n.byterange)
                    continue;
                const r = s.byterange
                  , a = n.byterange;
                if (r && !a || !r && a)
                    return !1;
                if (r.offset !== a.offset || r.length !== a.length)
                    return !1
            }
            return !0
        }
          , $r = (e,t,i,s)=>`placeholder-uri-${e}-${t}-${s.attributes.NAME || i}`
          , qr = (e,t,i)=>{
            let s = !0
              , n = Mn(e, {
                duration: t.duration,
                minimumUpdatePeriod: t.minimumUpdatePeriod,
                timelineStarts: t.timelineStarts
            });
            for (let r = 0; r < t.playlists.length; r++) {
                const e = t.playlists[r];
                if (e.sidx) {
                    const t = (0,
                    b.mm)(e.sidx);
                    i && i[t] && i[t].sidx && (0,
                    b.jp)(e, i[t].sidx, e.sidx.resolvedUri)
                }
                const a = br(n, e, jr);
                a && (n = a,
                s = !1)
            }
            return cr(t, ((e,t,i,r)=>{
                if (e.playlists && e.playlists.length) {
                    const a = e.playlists[0].id
                      , o = br(n, e.playlists[0], jr);
                    o && (n = o,
                    r in n.mediaGroups[t][i] || (n.mediaGroups[t][i][r] = e),
                    n.mediaGroups[t][i][r].playlists[0] = n.playlists[a],
                    s = !1)
                }
            }
            )),
            ((e,t)=>{
                cr(e, ((i,s,n,r)=>{
                    r in t.mediaGroups[s][n] || delete e.mediaGroups[s][n][r]
                }
                ))
            }
            )(n, t),
            t.minimumUpdatePeriod !== e.minimumUpdatePeriod && (s = !1),
            s ? null : n
        }
          , Hr = (e,t)=>{
            const i = {};
            for (const r in e) {
                const a = e[r].sidx;
                if (a) {
                    const e = (0,
                    b.mm)(a);
                    if (!t[e])
                        break;
                    const r = t[e].sidxInfo;
                    s = r,
                    n = a,
                    (Boolean(!s.map && !n.map) || Boolean(s.map && n.map && s.map.byterange.offset === n.map.byterange.offset && s.map.byterange.length === n.map.byterange.length)) && s.uri === n.uri && s.byterange.offset === n.byterange.offset && s.byterange.length === n.byterange.length && (i[e] = t[e])
                }
            }
            var s, n;
            return i
        }
        ;
        class Vr extends Fr {
            constructor(e, t, i={}, s) {
                super(),
                this.mainPlaylistLoader_ = s || this,
                s || (this.isMain_ = !0);
                const {withCredentials: n=!1} = i;
                if (this.vhs_ = t,
                this.withCredentials = n,
                this.addMetadataToTextTrack = i.addMetadataToTextTrack,
                !e)
                    throw new Error("A non-empty playlist URL or object is required");
                this.on("minimumUpdatePeriod", (()=>{
                    this.refreshXml_()
                }
                )),
                this.on("mediaupdatetimeout", (()=>{
                    this.refreshMedia_(this.media().id)
                }
                )),
                this.state = "HAVE_NOTHING",
                this.loadedPlaylists_ = {},
                this.logger_ = On("DashPlaylistLoader"),
                this.isMain_ ? (this.mainPlaylistLoader_.srcUrl = e,
                this.mainPlaylistLoader_.sidxMapping_ = {}) : this.childPlaylist_ = e
            }
            requestErrored_(e, t, i) {
                return !this.request || (this.request = null,
                e ? (this.error = "object" !== typeof e || e instanceof Error ? {
                    status: t.status,
                    message: "DASH request error at URL: " + t.uri,
                    response: t.response,
                    code: 2
                } : e,
                i && (this.state = i),
                this.trigger("error"),
                !0) : void 0)
            }
            addSidxSegments_(e, t, i) {
                const s = e.sidx && (0,
                b.mm)(e.sidx);
                if (!e.sidx || !s || this.mainPlaylistLoader_.sidxMapping_[s])
                    return void (this.mediaRequest_ = n().setTimeout((()=>i(!1)), 0));
                const r = Dn(e.sidx.resolvedUri)
                  , a = (n,r)=>{
                    if (this.requestErrored_(n, r, t))
                        return;
                    const a = this.mainPlaylistLoader_.sidxMapping_;
                    let o;
                    try {
                        o = k()((0,
                        T.Ki)(r.response).subarray(8))
                    } catch (l) {
                        return void this.requestErrored_(l, r, t)
                    }
                    return a[s] = {
                        sidxInfo: e.sidx,
                        sidx: o
                    },
                    (0,
                    b.jp)(e, o, e.sidx.resolvedUri),
                    i(!0)
                }
                ;
                this.request = ((e,t,i)=>{
                    let s, n = [], r = !1;
                    const a = function(e, t, s, n) {
                        return t.abort(),
                        r = !0,
                        i(e, t, s, n)
                    }
                      , o = function(e, t) {
                        if (r)
                            return;
                        if (e)
                            return a(e, t, "", n);
                        const i = t.responseText.substring(n && n.byteLength || 0, t.responseText.length);
                        if (n = (0,
                        T.lx)(n, (0,
                        T.qX)(i, !0)),
                        s = s || (0,
                        C.c)(n),
                        n.length < 10 || s && n.length < s + 2)
                            return Nr(t, (()=>a(e, t, "", n)));
                        const o = (0,
                        E.Xm)(n);
                        return "ts" === o && n.length < 188 || !o && n.length < 376 ? Nr(t, (()=>a(e, t, "", n))) : a(null, t, o, n)
                    }
                      , l = t({
                        uri: e,
                        beforeSend(e) {
                            e.overrideMimeType("text/plain; charset=x-user-defined"),
                            e.addEventListener("progress", (function({total: t, loaded: i}) {
                                return Er(e, null, {
                                    statusCode: e.status
                                }, o)
                            }
                            ))
                        }
                    }, (function(e, t) {
                        return Er(l, e, t, o)
                    }
                    ));
                    return l
                }
                )(r, this.vhs_.xhr, ((t,i,s,n)=>{
                    if (t)
                        return a(t, i);
                    if (!s || "mp4" !== s)
                        return a({
                            status: i.status,
                            message: `Unsupported ${s || "unknown"} container type for sidx segment at URL: ${r}`,
                            response: "",
                            playlist: e,
                            internal: !0,
                            playlistExclusionDuration: 1 / 0,
                            code: 2
                        }, i);
                    const {offset: o, length: l} = e.sidx.byterange;
                    if (n.length >= l + o)
                        return a(t, {
                            response: n.subarray(o, o + l),
                            status: i.status,
                            uri: i.uri
                        });
                    this.request = this.vhs_.xhr({
                        uri: r,
                        responseType: "arraybuffer",
                        headers: xr({
                            byterange: e.sidx.byterange
                        })
                    }, a)
                }
                ))
            }
            dispose() {
                this.trigger("dispose"),
                this.stopRequest(),
                this.loadedPlaylists_ = {},
                n().clearTimeout(this.minimumUpdatePeriodTimeout_),
                n().clearTimeout(this.mediaRequest_),
                n().clearTimeout(this.mediaUpdateTimeout),
                this.mediaUpdateTimeout = null,
                this.mediaRequest_ = null,
                this.minimumUpdatePeriodTimeout_ = null,
                this.mainPlaylistLoader_.createMupOnMedia_ && (this.off("loadedmetadata", this.mainPlaylistLoader_.createMupOnMedia_),
                this.mainPlaylistLoader_.createMupOnMedia_ = null),
                this.off()
            }
            hasPendingRequest() {
                return this.request || this.mediaRequest_
            }
            stopRequest() {
                if (this.request) {
                    const e = this.request;
                    this.request = null,
                    e.onreadystatechange = null,
                    e.abort()
                }
            }
            media(e) {
                if (!e)
                    return this.media_;
                if ("HAVE_NOTHING" === this.state)
                    throw new Error("Cannot switch media playlist from " + this.state);
                const t = this.state;
                if ("string" === typeof e) {
                    if (!this.mainPlaylistLoader_.main.playlists[e])
                        throw new Error("Unknown playlist URI: " + e);
                    e = this.mainPlaylistLoader_.main.playlists[e]
                }
                const i = !this.media_ || e.id !== this.media_.id;
                if (i && this.loadedPlaylists_[e.id] && this.loadedPlaylists_[e.id].endList)
                    return this.state = "HAVE_METADATA",
                    this.media_ = e,
                    void (i && (this.trigger("mediachanging"),
                    this.trigger("mediachange")));
                i && (this.media_ && this.trigger("mediachanging"),
                this.addSidxSegments_(e, t, (i=>{
                    this.haveMetadata({
                        startingState: t,
                        playlist: e
                    })
                }
                )))
            }
            haveMetadata({startingState: e, playlist: t}) {
                this.state = "HAVE_METADATA",
                this.loadedPlaylists_[t.id] = t,
                this.mediaRequest_ = null,
                this.refreshMedia_(t.id),
                "HAVE_MAIN_MANIFEST" === e ? this.trigger("loadedmetadata") : this.trigger("mediachange")
            }
            pause() {
                this.mainPlaylistLoader_.createMupOnMedia_ && (this.off("loadedmetadata", this.mainPlaylistLoader_.createMupOnMedia_),
                this.mainPlaylistLoader_.createMupOnMedia_ = null),
                this.stopRequest(),
                n().clearTimeout(this.mediaUpdateTimeout),
                this.mediaUpdateTimeout = null,
                this.isMain_ && (n().clearTimeout(this.mainPlaylistLoader_.minimumUpdatePeriodTimeout_),
                this.mainPlaylistLoader_.minimumUpdatePeriodTimeout_ = null),
                "HAVE_NOTHING" === this.state && (this.started = !1)
            }
            load(e) {
                n().clearTimeout(this.mediaUpdateTimeout),
                this.mediaUpdateTimeout = null;
                const t = this.media();
                if (e) {
                    const e = t ? t.targetDuration / 2 * 1e3 : 5e3;
                    this.mediaUpdateTimeout = n().setTimeout((()=>this.load()), e)
                } else
                    this.started ? t && !t.endList ? (this.isMain_ && !this.minimumUpdatePeriodTimeout_ && (this.trigger("minimumUpdatePeriod"),
                    this.updateMinimumUpdatePeriodTimeout_()),
                    this.trigger("mediaupdatetimeout")) : this.trigger("loadedplaylist") : this.start()
            }
            start() {
                this.started = !0,
                this.isMain_ ? this.requestMain_(((e,t)=>{
                    this.haveMain_(),
                    this.hasPendingRequest() || this.media_ || this.media(this.mainPlaylistLoader_.main.playlists[0])
                }
                )) : this.mediaRequest_ = n().setTimeout((()=>this.haveMain_()), 0)
            }
            requestMain_(e) {
                this.request = this.vhs_.xhr({
                    uri: this.mainPlaylistLoader_.srcUrl,
                    withCredentials: this.withCredentials
                }, ((t,i)=>{
                    if (this.requestErrored_(t, i))
                        return void ("HAVE_NOTHING" === this.state && (this.started = !1));
                    const s = i.responseText !== this.mainPlaylistLoader_.mainXml_;
                    return this.mainPlaylistLoader_.mainXml_ = i.responseText,
                    i.responseHeaders && i.responseHeaders.date ? this.mainLoaded_ = Date.parse(i.responseHeaders.date) : this.mainLoaded_ = Date.now(),
                    this.mainPlaylistLoader_.srcUrl = Dn(this.mainPlaylistLoader_.srcUrl, i),
                    s ? (this.handleMain_(),
                    void this.syncClientServerClock_((()=>e(i, s)))) : e(i, s)
                }
                ))
            }
            syncClientServerClock_(e) {
                const t = (0,
                b.LG)(this.mainPlaylistLoader_.mainXml_);
                return null === t ? (this.mainPlaylistLoader_.clientOffset_ = this.mainLoaded_ - Date.now(),
                e()) : "DIRECT" === t.method ? (this.mainPlaylistLoader_.clientOffset_ = t.value - Date.now(),
                e()) : void (this.request = this.vhs_.xhr({
                    uri: Ln(this.mainPlaylistLoader_.srcUrl, t.value),
                    method: t.method,
                    withCredentials: this.withCredentials
                }, ((i,s)=>{
                    if (!this.request)
                        return;
                    if (i)
                        return this.mainPlaylistLoader_.clientOffset_ = this.mainLoaded_ - Date.now(),
                        e();
                    let n;
                    n = "HEAD" === t.method ? s.responseHeaders && s.responseHeaders.date ? Date.parse(s.responseHeaders.date) : this.mainLoaded_ : Date.parse(s.responseText),
                    this.mainPlaylistLoader_.clientOffset_ = n - Date.now(),
                    e()
                }
                )))
            }
            haveMain_() {
                this.state = "HAVE_MAIN_MANIFEST",
                this.isMain_ ? this.trigger("loadedplaylist") : this.media_ || this.media(this.childPlaylist_)
            }
            handleMain_() {
                this.mediaRequest_ = null;
                const e = this.mainPlaylistLoader_.main;
                let t = (({mainXml: e, srcUrl: t, clientOffset: i, sidxMapping: s, previousManifest: n})=>{
                    const r = (0,
                    b.Qc)(e, {
                        manifestUri: t,
                        clientOffset: i,
                        sidxMapping: s,
                        previousManifest: n
                    });
                    return mr(r, t, $r),
                    r
                }
                )({
                    mainXml: this.mainPlaylistLoader_.mainXml_,
                    srcUrl: this.mainPlaylistLoader_.srcUrl,
                    clientOffset: this.mainPlaylistLoader_.clientOffset_,
                    sidxMapping: this.mainPlaylistLoader_.sidxMapping_,
                    previousManifest: e
                });
                e && (t = qr(e, t, this.mainPlaylistLoader_.sidxMapping_)),
                this.mainPlaylistLoader_.main = t || e;
                const i = this.mainPlaylistLoader_.main.locations && this.mainPlaylistLoader_.main.locations[0];
                return i && i !== this.mainPlaylistLoader_.srcUrl && (this.mainPlaylistLoader_.srcUrl = i),
                (!e || t && t.minimumUpdatePeriod !== e.minimumUpdatePeriod) && this.updateMinimumUpdatePeriodTimeout_(),
                this.addEventStreamToMetadataTrack_(t),
                Boolean(t)
            }
            updateMinimumUpdatePeriodTimeout_() {
                const e = this.mainPlaylistLoader_;
                e.createMupOnMedia_ && (e.off("loadedmetadata", e.createMupOnMedia_),
                e.createMupOnMedia_ = null),
                e.minimumUpdatePeriodTimeout_ && (n().clearTimeout(e.minimumUpdatePeriodTimeout_),
                e.minimumUpdatePeriodTimeout_ = null);
                let t = e.main && e.main.minimumUpdatePeriod;
                0 === t && (e.media() ? t = 1e3 * e.media().targetDuration : (e.createMupOnMedia_ = e.updateMinimumUpdatePeriodTimeout_,
                e.one("loadedmetadata", e.createMupOnMedia_))),
                "number" !== typeof t || t <= 0 ? t < 0 && this.logger_(`found invalid minimumUpdatePeriod of ${t}, not setting a timeout`) : this.createMUPTimeout_(t)
            }
            createMUPTimeout_(e) {
                const t = this.mainPlaylistLoader_;
                t.minimumUpdatePeriodTimeout_ = n().setTimeout((()=>{
                    t.minimumUpdatePeriodTimeout_ = null,
                    t.trigger("minimumUpdatePeriod"),
                    t.createMUPTimeout_(e)
                }
                ), e)
            }
            refreshXml_() {
                this.requestMain_(((e,t)=>{
                    t && (this.media_ && (this.media_ = this.mainPlaylistLoader_.main.playlists[this.media_.id]),
                    this.mainPlaylistLoader_.sidxMapping_ = ((e,t)=>{
                        let i = Hr(e.playlists, t);
                        return cr(e, ((e,s,n,r)=>{
                            if (e.playlists && e.playlists.length) {
                                const s = e.playlists;
                                i = Mn(i, Hr(s, t))
                            }
                        }
                        )),
                        i
                    }
                    )(this.mainPlaylistLoader_.main, this.mainPlaylistLoader_.sidxMapping_),
                    this.addSidxSegments_(this.media(), this.state, (e=>{
                        this.refreshMedia_(this.media().id)
                    }
                    )))
                }
                ))
            }
            refreshMedia_(e) {
                if (!e)
                    throw new Error("refreshMedia_ must take a media id");
                this.media_ && this.isMain_ && this.handleMain_();
                const t = this.mainPlaylistLoader_.main.playlists
                  , i = !this.media_ || this.media_ !== t[e];
                if (i ? this.media_ = t[e] : this.trigger("playlistunchanged"),
                !this.mediaUpdateTimeout) {
                    const e = ()=>{
                        this.media().endList || (this.mediaUpdateTimeout = n().setTimeout((()=>{
                            this.trigger("mediaupdatetimeout"),
                            e()
                        }
                        ), Sr(this.media(), Boolean(i))))
                    }
                    ;
                    e()
                }
                this.trigger("loadedplaylist")
            }
            addEventStreamToMetadataTrack_(e) {
                if (e && this.mainPlaylistLoader_.main.eventStream) {
                    const e = this.mainPlaylistLoader_.main.eventStream.map((e=>({
                        cueTime: e.start,
                        frames: [{
                            data: e.messageData
                        }]
                    })));
                    this.addMetadataToTextTrack("EventStream", e, this.mainPlaylistLoader_.main.duration)
                }
            }
            getKeyIdSet(e) {
                if (e.contentProtection) {
                    const t = new Set;
                    for (const i in e.contentProtection) {
                        const s = e.contentProtection[i].attributes["cenc:default_KID"];
                        s && t.add(s.replace(/-/g, "").toLowerCase())
                    }
                    return t
                }
            }
        }
        var zr = {
            GOAL_BUFFER_LENGTH: 30,
            MAX_GOAL_BUFFER_LENGTH: 60,
            BACK_BUFFER_LENGTH: 30,
            GOAL_BUFFER_LENGTH_RATE: 1,
            INITIAL_BANDWIDTH: 4194304,
            BANDWIDTH_VARIANCE: 1.2,
            BUFFER_LOW_WATER_LINE: 0,
            MAX_BUFFER_LOW_WATER_LINE: 30,
            EXPERIMENTAL_MAX_BUFFER_LOW_WATER_LINE: 16,
            BUFFER_LOW_WATER_LINE_RATE: 1,
            BUFFER_HIGH_WATER_LINE: 30
        };
        const Wr = function(e) {
            return e.on = e.addEventListener,
            e.off = e.removeEventListener,
            e
        }
          , Gr = function(e) {
            return function() {
                const t = function(e) {
                    try {
                        return URL.createObjectURL(new Blob([e],{
                            type: "application/javascript"
                        }))
                    } catch (t) {
                        const i = new BlobBuilder;
                        return i.append(e),
                        URL.createObjectURL(i.getBlob())
                    }
                }(e)
                  , i = Wr(new Worker(t));
                i.objURL = t;
                const s = i.terminate;
                return i.on = i.addEventListener,
                i.off = i.removeEventListener,
                i.terminate = function() {
                    return URL.revokeObjectURL(t),
                    s.call(this)
                }
                ,
                i
            }
        }
          , Kr = function(e) {
            return `var browserWorkerPolyFill = ${Wr.toString()};\nbrowserWorkerPolyFill(self);\n` + e
        }
          , Qr = function(e) {
            return e.toString().replace(/^function.+?{/, "").slice(0, -1)
        };
        var Xr = Gr(Kr(Qr((function() {
            var e = "undefined" !== typeof globalThis ? globalThis : "undefined" !== typeof window ? window : "undefined" !== typeof i.g ? i.g : "undefined" !== typeof self ? self : {}
              , t = function() {
                this.init = function() {
                    var e = {};
                    this.on = function(t, i) {
                        e[t] || (e[t] = []),
                        e[t] = e[t].concat(i)
                    }
                    ,
                    this.off = function(t, i) {
                        var s;
                        return !!e[t] && (s = e[t].indexOf(i),
                        e[t] = e[t].slice(),
                        e[t].splice(s, 1),
                        s > -1)
                    }
                    ,
                    this.trigger = function(t) {
                        var i, s, n, r;
                        if (i = e[t])
                            if (2 === arguments.length)
                                for (n = i.length,
                                s = 0; s < n; ++s)
                                    i[s].call(this, arguments[1]);
                            else {
                                for (r = [],
                                s = arguments.length,
                                s = 1; s < arguments.length; ++s)
                                    r.push(arguments[s]);
                                for (n = i.length,
                                s = 0; s < n; ++s)
                                    i[s].apply(this, r)
                            }
                    }
                    ,
                    this.dispose = function() {
                        e = {}
                    }
                }
            };
            t.prototype.pipe = function(e) {
                return this.on("data", (function(t) {
                    e.push(t)
                }
                )),
                this.on("done", (function(t) {
                    e.flush(t)
                }
                )),
                this.on("partialdone", (function(t) {
                    e.partialFlush(t)
                }
                )),
                this.on("endedtimeline", (function(t) {
                    e.endTimeline(t)
                }
                )),
                this.on("reset", (function(t) {
                    e.reset(t)
                }
                )),
                e
            }
            ,
            t.prototype.push = function(e) {
                this.trigger("data", e)
            }
            ,
            t.prototype.flush = function(e) {
                this.trigger("done", e)
            }
            ,
            t.prototype.partialFlush = function(e) {
                this.trigger("partialdone", e)
            }
            ,
            t.prototype.endTimeline = function(e) {
                this.trigger("endedtimeline", e)
            }
            ,
            t.prototype.reset = function(e) {
                this.trigger("reset", e)
            }
            ;
            var s, n, r, a, o, l, h, d, u, c, p, m, g, f, _, y, v, T, b, S, k, C, E, w, x, I, P, A, L, D, O, M, R, U, B, N, F = t, j = Math.pow(2, 32), $ = {
                getUint64: function(e) {
                    var t, i = new DataView(e.buffer,e.byteOffset,e.byteLength);
                    return i.getBigUint64 ? (t = i.getBigUint64(0)) < Number.MAX_SAFE_INTEGER ? Number(t) : t : i.getUint32(0) * j + i.getUint32(4)
                },
                MAX_UINT32: j
            }, q = $.MAX_UINT32;
            !function() {
                var e;
                if (E = {
                    avc1: [],
                    avcC: [],
                    btrt: [],
                    dinf: [],
                    dref: [],
                    esds: [],
                    ftyp: [],
                    hdlr: [],
                    mdat: [],
                    mdhd: [],
                    mdia: [],
                    mfhd: [],
                    minf: [],
                    moof: [],
                    moov: [],
                    mp4a: [],
                    mvex: [],
                    mvhd: [],
                    pasp: [],
                    sdtp: [],
                    smhd: [],
                    stbl: [],
                    stco: [],
                    stsc: [],
                    stsd: [],
                    stsz: [],
                    stts: [],
                    styp: [],
                    tfdt: [],
                    tfhd: [],
                    traf: [],
                    trak: [],
                    trun: [],
                    trex: [],
                    tkhd: [],
                    vmhd: []
                },
                "undefined" !== typeof Uint8Array) {
                    for (e in E)
                        E.hasOwnProperty(e) && (E[e] = [e.charCodeAt(0), e.charCodeAt(1), e.charCodeAt(2), e.charCodeAt(3)]);
                    w = new Uint8Array(["i".charCodeAt(0), "s".charCodeAt(0), "o".charCodeAt(0), "m".charCodeAt(0)]),
                    I = new Uint8Array(["a".charCodeAt(0), "v".charCodeAt(0), "c".charCodeAt(0), "1".charCodeAt(0)]),
                    x = new Uint8Array([0, 0, 0, 1]),
                    P = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0, 118, 105, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 105, 100, 101, 111, 72, 97, 110, 100, 108, 101, 114, 0]),
                    A = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0, 115, 111, 117, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 111, 117, 110, 100, 72, 97, 110, 100, 108, 101, 114, 0]),
                    L = {
                        video: P,
                        audio: A
                    },
                    M = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 117, 114, 108, 32, 0, 0, 0, 1]),
                    O = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0]),
                    R = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0]),
                    U = R,
                    B = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    N = R,
                    D = new Uint8Array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
                }
            }(),
            s = function(e) {
                var t, i, s = [], n = 0;
                for (t = 1; t < arguments.length; t++)
                    s.push(arguments[t]);
                for (t = s.length; t--; )
                    n += s[t].byteLength;
                for (i = new Uint8Array(n + 8),
                new DataView(i.buffer,i.byteOffset,i.byteLength).setUint32(0, i.byteLength),
                i.set(e, 4),
                t = 0,
                n = 8; t < s.length; t++)
                    i.set(s[t], n),
                    n += s[t].byteLength;
                return i
            }
            ,
            n = function() {
                return s(E.dinf, s(E.dref, M))
            }
            ,
            r = function(e) {
                return s(E.esds, new Uint8Array([0, 0, 0, 0, 3, 25, 0, 0, 0, 4, 17, 64, 21, 0, 6, 0, 0, 0, 218, 192, 0, 0, 218, 192, 5, 2, e.audioobjecttype << 3 | e.samplingfrequencyindex >>> 1, e.samplingfrequencyindex << 7 | e.channelcount << 3, 6, 1, 2]))
            }
            ,
            a = function() {
                return s(E.ftyp, w, x, w, I)
            }
            ,
            y = function(e) {
                return s(E.hdlr, L[e])
            }
            ,
            o = function(e) {
                return s(E.mdat, e)
            }
            ,
            _ = function(e) {
                var t = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 1, 95, 144, e.duration >>> 24 & 255, e.duration >>> 16 & 255, e.duration >>> 8 & 255, 255 & e.duration, 85, 196, 0, 0]);
                return e.samplerate && (t[12] = e.samplerate >>> 24 & 255,
                t[13] = e.samplerate >>> 16 & 255,
                t[14] = e.samplerate >>> 8 & 255,
                t[15] = 255 & e.samplerate),
                s(E.mdhd, t)
            }
            ,
            f = function(e) {
                return s(E.mdia, _(e), y(e.type), h(e))
            }
            ,
            l = function(e) {
                return s(E.mfhd, new Uint8Array([0, 0, 0, 0, (4278190080 & e) >> 24, (16711680 & e) >> 16, (65280 & e) >> 8, 255 & e]))
            }
            ,
            h = function(e) {
                return s(E.minf, "video" === e.type ? s(E.vmhd, D) : s(E.smhd, O), n(), T(e))
            }
            ,
            d = function(e, t) {
                for (var i = [], n = t.length; n--; )
                    i[n] = S(t[n]);
                return s.apply(null, [E.moof, l(e)].concat(i))
            }
            ,
            u = function(e) {
                for (var t = e.length, i = []; t--; )
                    i[t] = m(e[t]);
                return s.apply(null, [E.moov, p(4294967295)].concat(i).concat(c(e)))
            }
            ,
            c = function(e) {
                for (var t = e.length, i = []; t--; )
                    i[t] = k(e[t]);
                return s.apply(null, [E.mvex].concat(i))
            }
            ,
            p = function(e) {
                var t = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 1, 95, 144, (4278190080 & e) >> 24, (16711680 & e) >> 16, (65280 & e) >> 8, 255 & e, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255]);
                return s(E.mvhd, t)
            }
            ,
            v = function(e) {
                var t, i, n = e.samples || [], r = new Uint8Array(4 + n.length);
                for (i = 0; i < n.length; i++)
                    t = n[i].flags,
                    r[i + 4] = t.dependsOn << 4 | t.isDependedOn << 2 | t.hasRedundancy;
                return s(E.sdtp, r)
            }
            ,
            T = function(e) {
                return s(E.stbl, b(e), s(E.stts, N), s(E.stsc, U), s(E.stsz, B), s(E.stco, R))
            }
            ,
            function() {
                var e, t;
                b = function(i) {
                    return s(E.stsd, new Uint8Array([0, 0, 0, 0, 0, 0, 0, 1]), "video" === i.type ? e(i) : t(i))
                }
                ,
                e = function(e) {
                    var t, i, n = e.sps || [], r = e.pps || [], a = [], o = [];
                    for (t = 0; t < n.length; t++)
                        a.push((65280 & n[t].byteLength) >>> 8),
                        a.push(255 & n[t].byteLength),
                        a = a.concat(Array.prototype.slice.call(n[t]));
                    for (t = 0; t < r.length; t++)
                        o.push((65280 & r[t].byteLength) >>> 8),
                        o.push(255 & r[t].byteLength),
                        o = o.concat(Array.prototype.slice.call(r[t]));
                    if (i = [E.avc1, new Uint8Array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (65280 & e.width) >> 8, 255 & e.width, (65280 & e.height) >> 8, 255 & e.height, 0, 72, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 1, 19, 118, 105, 100, 101, 111, 106, 115, 45, 99, 111, 110, 116, 114, 105, 98, 45, 104, 108, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 17, 17]), s(E.avcC, new Uint8Array([1, e.profileIdc, e.profileCompatibility, e.levelIdc, 255].concat([n.length], a, [r.length], o))), s(E.btrt, new Uint8Array([0, 28, 156, 128, 0, 45, 198, 192, 0, 45, 198, 192]))],
                    e.sarRatio) {
                        var l = e.sarRatio[0]
                          , h = e.sarRatio[1];
                        i.push(s(E.pasp, new Uint8Array([(4278190080 & l) >> 24, (16711680 & l) >> 16, (65280 & l) >> 8, 255 & l, (4278190080 & h) >> 24, (16711680 & h) >> 16, (65280 & h) >> 8, 255 & h])))
                    }
                    return s.apply(null, i)
                }
                ,
                t = function(e) {
                    return s(E.mp4a, new Uint8Array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, (65280 & e.channelcount) >> 8, 255 & e.channelcount, (65280 & e.samplesize) >> 8, 255 & e.samplesize, 0, 0, 0, 0, (65280 & e.samplerate) >> 8, 255 & e.samplerate, 0, 0]), r(e))
                }
            }(),
            g = function(e) {
                var t = new Uint8Array([0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, (4278190080 & e.id) >> 24, (16711680 & e.id) >> 16, (65280 & e.id) >> 8, 255 & e.id, 0, 0, 0, 0, (4278190080 & e.duration) >> 24, (16711680 & e.duration) >> 16, (65280 & e.duration) >> 8, 255 & e.duration, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, (65280 & e.width) >> 8, 255 & e.width, 0, 0, (65280 & e.height) >> 8, 255 & e.height, 0, 0]);
                return s(E.tkhd, t)
            }
            ,
            S = function(e) {
                var t, i, n, r, a, o;
                return t = s(E.tfhd, new Uint8Array([0, 0, 0, 58, (4278190080 & e.id) >> 24, (16711680 & e.id) >> 16, (65280 & e.id) >> 8, 255 & e.id, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
                a = Math.floor(e.baseMediaDecodeTime / q),
                o = Math.floor(e.baseMediaDecodeTime % q),
                i = s(E.tfdt, new Uint8Array([1, 0, 0, 0, a >>> 24 & 255, a >>> 16 & 255, a >>> 8 & 255, 255 & a, o >>> 24 & 255, o >>> 16 & 255, o >>> 8 & 255, 255 & o])),
                92,
                "audio" === e.type ? (n = C(e, 92),
                s(E.traf, t, i, n)) : (r = v(e),
                n = C(e, r.length + 92),
                s(E.traf, t, i, n, r))
            }
            ,
            m = function(e) {
                return e.duration = e.duration || 4294967295,
                s(E.trak, g(e), f(e))
            }
            ,
            k = function(e) {
                var t = new Uint8Array([0, 0, 0, 0, (4278190080 & e.id) >> 24, (16711680 & e.id) >> 16, (65280 & e.id) >> 8, 255 & e.id, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]);
                return "video" !== e.type && (t[t.length - 1] = 0),
                s(E.trex, t)
            }
            ,
            function() {
                var e, t, i;
                i = function(e, t) {
                    var i = 0
                      , s = 0
                      , n = 0
                      , r = 0;
                    return e.length && (void 0 !== e[0].duration && (i = 1),
                    void 0 !== e[0].size && (s = 2),
                    void 0 !== e[0].flags && (n = 4),
                    void 0 !== e[0].compositionTimeOffset && (r = 8)),
                    [0, 0, i | s | n | r, 1, (4278190080 & e.length) >>> 24, (16711680 & e.length) >>> 16, (65280 & e.length) >>> 8, 255 & e.length, (4278190080 & t) >>> 24, (16711680 & t) >>> 16, (65280 & t) >>> 8, 255 & t]
                }
                ,
                t = function(e, t) {
                    var n, r, a, o, l, h;
                    for (t += 20 + 16 * (o = e.samples || []).length,
                    a = i(o, t),
                    (r = new Uint8Array(a.length + 16 * o.length)).set(a),
                    n = a.length,
                    h = 0; h < o.length; h++)
                        l = o[h],
                        r[n++] = (4278190080 & l.duration) >>> 24,
                        r[n++] = (16711680 & l.duration) >>> 16,
                        r[n++] = (65280 & l.duration) >>> 8,
                        r[n++] = 255 & l.duration,
                        r[n++] = (4278190080 & l.size) >>> 24,
                        r[n++] = (16711680 & l.size) >>> 16,
                        r[n++] = (65280 & l.size) >>> 8,
                        r[n++] = 255 & l.size,
                        r[n++] = l.flags.isLeading << 2 | l.flags.dependsOn,
                        r[n++] = l.flags.isDependedOn << 6 | l.flags.hasRedundancy << 4 | l.flags.paddingValue << 1 | l.flags.isNonSyncSample,
                        r[n++] = 61440 & l.flags.degradationPriority,
                        r[n++] = 15 & l.flags.degradationPriority,
                        r[n++] = (4278190080 & l.compositionTimeOffset) >>> 24,
                        r[n++] = (16711680 & l.compositionTimeOffset) >>> 16,
                        r[n++] = (65280 & l.compositionTimeOffset) >>> 8,
                        r[n++] = 255 & l.compositionTimeOffset;
                    return s(E.trun, r)
                }
                ,
                e = function(e, t) {
                    var n, r, a, o, l, h;
                    for (t += 20 + 8 * (o = e.samples || []).length,
                    a = i(o, t),
                    (n = new Uint8Array(a.length + 8 * o.length)).set(a),
                    r = a.length,
                    h = 0; h < o.length; h++)
                        l = o[h],
                        n[r++] = (4278190080 & l.duration) >>> 24,
                        n[r++] = (16711680 & l.duration) >>> 16,
                        n[r++] = (65280 & l.duration) >>> 8,
                        n[r++] = 255 & l.duration,
                        n[r++] = (4278190080 & l.size) >>> 24,
                        n[r++] = (16711680 & l.size) >>> 16,
                        n[r++] = (65280 & l.size) >>> 8,
                        n[r++] = 255 & l.size;
                    return s(E.trun, n)
                }
                ,
                C = function(i, s) {
                    return "audio" === i.type ? e(i, s) : t(i, s)
                }
            }();
            var H, V, z, W, G, K, Q, X, Y = {
                ftyp: a,
                mdat: o,
                moof: d,
                moov: u,
                initSegment: function(e) {
                    var t, i = a(), s = u(e);
                    return (t = new Uint8Array(i.byteLength + s.byteLength)).set(i),
                    t.set(s, i.byteLength),
                    t
                }
            }, J = function(e, t) {
                var i = {
                    size: 0,
                    flags: {
                        isLeading: 0,
                        dependsOn: 1,
                        isDependedOn: 0,
                        hasRedundancy: 0,
                        degradationPriority: 0,
                        isNonSyncSample: 1
                    }
                };
                return i.dataOffset = t,
                i.compositionTimeOffset = e.pts - e.dts,
                i.duration = e.duration,
                i.size = 4 * e.length,
                i.size += e.byteLength,
                e.keyFrame && (i.flags.dependsOn = 2,
                i.flags.isNonSyncSample = 0),
                i
            }, Z = {
                groupNalsIntoFrames: function(e) {
                    var t, i, s = [], n = [];
                    for (n.byteLength = 0,
                    n.nalCount = 0,
                    n.duration = 0,
                    s.byteLength = 0,
                    t = 0; t < e.length; t++)
                        "access_unit_delimiter_rbsp" === (i = e[t]).nalUnitType ? (s.length && (s.duration = i.dts - s.dts,
                        n.byteLength += s.byteLength,
                        n.nalCount += s.length,
                        n.duration += s.duration,
                        n.push(s)),
                        (s = [i]).byteLength = i.data.byteLength,
                        s.pts = i.pts,
                        s.dts = i.dts) : ("slice_layer_without_partitioning_rbsp_idr" === i.nalUnitType && (s.keyFrame = !0),
                        s.duration = i.dts - s.dts,
                        s.byteLength += i.data.byteLength,
                        s.push(i));
                    return n.length && (!s.duration || s.duration <= 0) && (s.duration = n[n.length - 1].duration),
                    n.byteLength += s.byteLength,
                    n.nalCount += s.length,
                    n.duration += s.duration,
                    n.push(s),
                    n
                },
                groupFramesIntoGops: function(e) {
                    var t, i, s = [], n = [];
                    for (s.byteLength = 0,
                    s.nalCount = 0,
                    s.duration = 0,
                    s.pts = e[0].pts,
                    s.dts = e[0].dts,
                    n.byteLength = 0,
                    n.nalCount = 0,
                    n.duration = 0,
                    n.pts = e[0].pts,
                    n.dts = e[0].dts,
                    t = 0; t < e.length; t++)
                        (i = e[t]).keyFrame ? (s.length && (n.push(s),
                        n.byteLength += s.byteLength,
                        n.nalCount += s.nalCount,
                        n.duration += s.duration),
                        (s = [i]).nalCount = i.length,
                        s.byteLength = i.byteLength,
                        s.pts = i.pts,
                        s.dts = i.dts,
                        s.duration = i.duration) : (s.duration += i.duration,
                        s.nalCount += i.length,
                        s.byteLength += i.byteLength,
                        s.push(i));
                    return n.length && s.duration <= 0 && (s.duration = n[n.length - 1].duration),
                    n.byteLength += s.byteLength,
                    n.nalCount += s.nalCount,
                    n.duration += s.duration,
                    n.push(s),
                    n
                },
                extendFirstKeyFrame: function(e) {
                    var t;
                    return !e[0][0].keyFrame && e.length > 1 && (t = e.shift(),
                    e.byteLength -= t.byteLength,
                    e.nalCount -= t.nalCount,
                    e[0][0].dts = t.dts,
                    e[0][0].pts = t.pts,
                    e[0][0].duration += t.duration),
                    e
                },
                generateSampleTable: function(e, t) {
                    var i, s, n, r, a, o = t || 0, l = [];
                    for (i = 0; i < e.length; i++)
                        for (r = e[i],
                        s = 0; s < r.length; s++)
                            a = r[s],
                            o += (n = J(a, o)).size,
                            l.push(n);
                    return l
                },
                concatenateNalData: function(e) {
                    var t, i, s, n, r, a, o = 0, l = e.byteLength, h = e.nalCount, d = new Uint8Array(l + 4 * h), u = new DataView(d.buffer);
                    for (t = 0; t < e.length; t++)
                        for (n = e[t],
                        i = 0; i < n.length; i++)
                            for (r = n[i],
                            s = 0; s < r.length; s++)
                                a = r[s],
                                u.setUint32(o, a.data.byteLength),
                                o += 4,
                                d.set(a.data, o),
                                o += a.data.byteLength;
                    return d
                },
                generateSampleTableForFrame: function(e, t) {
                    var i, s = [];
                    return i = J(e, t || 0),
                    s.push(i),
                    s
                },
                concatenateNalDataForFrame: function(e) {
                    var t, i, s = 0, n = e.byteLength, r = e.length, a = new Uint8Array(n + 4 * r), o = new DataView(a.buffer);
                    for (t = 0; t < e.length; t++)
                        i = e[t],
                        o.setUint32(s, i.data.byteLength),
                        s += 4,
                        a.set(i.data, s),
                        s += i.data.byteLength;
                    return a
                }
            }, ee = [33, 16, 5, 32, 164, 27], te = [33, 65, 108, 84, 1, 2, 4, 8, 168, 2, 4, 8, 17, 191, 252], ie = function(e) {
                for (var t = []; e--; )
                    t.push(0);
                return t
            }, se = 9e4;
            K = function(e, t) {
                return V(G(e, t))
            }
            ,
            Q = function(e, t) {
                return z(W(e), t)
            }
            ,
            X = function(e, t, i) {
                return W(i ? e : e - t)
            }
            ;
            var ne = {
                ONE_SECOND_IN_TS: se,
                secondsToVideoTs: V = function(e) {
                    return e * se
                }
                ,
                secondsToAudioTs: z = function(e, t) {
                    return e * t
                }
                ,
                videoTsToSeconds: W = function(e) {
                    return e / se
                }
                ,
                audioTsToSeconds: G = function(e, t) {
                    return e / t
                }
                ,
                audioTsToVideoTs: K,
                videoTsToAudioTs: Q,
                metadataTsToSeconds: X
            }
              , re = function() {
                if (!H) {
                    var e = {
                        96e3: [ee, [227, 64], ie(154), [56]],
                        88200: [ee, [231], ie(170), [56]],
                        64e3: [ee, [248, 192], ie(240), [56]],
                        48e3: [ee, [255, 192], ie(268), [55, 148, 128], ie(54), [112]],
                        44100: [ee, [255, 192], ie(268), [55, 163, 128], ie(84), [112]],
                        32e3: [ee, [255, 192], ie(268), [55, 234], ie(226), [112]],
                        24e3: [ee, [255, 192], ie(268), [55, 255, 128], ie(268), [111, 112], ie(126), [224]],
                        16e3: [ee, [255, 192], ie(268), [55, 255, 128], ie(268), [111, 255], ie(269), [223, 108], ie(195), [1, 192]],
                        12e3: [te, ie(268), [3, 127, 248], ie(268), [6, 255, 240], ie(268), [13, 255, 224], ie(268), [27, 253, 128], ie(259), [56]],
                        11025: [te, ie(268), [3, 127, 248], ie(268), [6, 255, 240], ie(268), [13, 255, 224], ie(268), [27, 255, 192], ie(268), [55, 175, 128], ie(108), [112]],
                        8e3: [te, ie(268), [3, 121, 16], ie(47), [7]]
                    };
                    t = e,
                    H = Object.keys(t).reduce((function(e, i) {
                        return e[i] = new Uint8Array(t[i].reduce((function(e, t) {
                            return e.concat(t)
                        }
                        ), [])),
                        e
                    }
                    ), {})
                }
                var t;
                return H
            }
              , ae = ne
              , oe = {
                prefixWithSilence: function(e, t, i, s) {
                    var n, r, a, o, l, h = 0, d = 0, u = 0;
                    if (t.length && (n = ae.audioTsToVideoTs(e.baseMediaDecodeTime, e.samplerate),
                    r = Math.ceil(ae.ONE_SECOND_IN_TS / (e.samplerate / 1024)),
                    i && s && (h = n - Math.max(i, s),
                    u = (d = Math.floor(h / r)) * r),
                    !(d < 1 || u > ae.ONE_SECOND_IN_TS / 2))) {
                        for ((a = re()[e.samplerate]) || (a = t[0].data),
                        o = 0; o < d; o++)
                            l = t[0],
                            t.splice(0, 0, {
                                data: a,
                                dts: l.dts - r,
                                pts: l.pts - r
                            });
                        return e.baseMediaDecodeTime -= Math.floor(ae.videoTsToAudioTs(u, e.samplerate)),
                        u
                    }
                },
                trimAdtsFramesByEarliestDts: function(e, t, i) {
                    return t.minSegmentDts >= i ? e : (t.minSegmentDts = 1 / 0,
                    e.filter((function(e) {
                        return e.dts >= i && (t.minSegmentDts = Math.min(t.minSegmentDts, e.dts),
                        t.minSegmentPts = t.minSegmentDts,
                        !0)
                    }
                    )))
                },
                generateSampleTable: function(e) {
                    var t, i, s = [];
                    for (t = 0; t < e.length; t++)
                        i = e[t],
                        s.push({
                            size: i.data.byteLength,
                            duration: 1024
                        });
                    return s
                },
                concatenateFrameData: function(e) {
                    var t, i, s = 0, n = new Uint8Array(function(e) {
                        var t, i = 0;
                        for (t = 0; t < e.length; t++)
                            i += e[t].data.byteLength;
                        return i
                    }(e));
                    for (t = 0; t < e.length; t++)
                        i = e[t],
                        n.set(i.data, s),
                        s += i.data.byteLength;
                    return n
                }
            }
              , le = ne.ONE_SECOND_IN_TS
              , he = {
                clearDtsInfo: function(e) {
                    delete e.minSegmentDts,
                    delete e.maxSegmentDts,
                    delete e.minSegmentPts,
                    delete e.maxSegmentPts
                },
                calculateTrackBaseMediaDecodeTime: function(e, t) {
                    var i, s = e.minSegmentDts;
                    return t || (s -= e.timelineStartInfo.dts),
                    i = e.timelineStartInfo.baseMediaDecodeTime,
                    i += s,
                    i = Math.max(0, i),
                    "audio" === e.type && (i *= e.samplerate / le,
                    i = Math.floor(i)),
                    i
                },
                collectDtsInfo: function(e, t) {
                    "number" === typeof t.pts && (void 0 === e.timelineStartInfo.pts && (e.timelineStartInfo.pts = t.pts),
                    void 0 === e.minSegmentPts ? e.minSegmentPts = t.pts : e.minSegmentPts = Math.min(e.minSegmentPts, t.pts),
                    void 0 === e.maxSegmentPts ? e.maxSegmentPts = t.pts : e.maxSegmentPts = Math.max(e.maxSegmentPts, t.pts)),
                    "number" === typeof t.dts && (void 0 === e.timelineStartInfo.dts && (e.timelineStartInfo.dts = t.dts),
                    void 0 === e.minSegmentDts ? e.minSegmentDts = t.dts : e.minSegmentDts = Math.min(e.minSegmentDts, t.dts),
                    void 0 === e.maxSegmentDts ? e.maxSegmentDts = t.dts : e.maxSegmentDts = Math.max(e.maxSegmentDts, t.dts))
                }
            }
              , de = {
                parseSei: function(e) {
                    for (var t = 0, i = {
                        payloadType: -1,
                        payloadSize: 0
                    }, s = 0, n = 0; t < e.byteLength && 128 !== e[t]; ) {
                        for (; 255 === e[t]; )
                            s += 255,
                            t++;
                        for (s += e[t++]; 255 === e[t]; )
                            n += 255,
                            t++;
                        if (n += e[t++],
                        !i.payload && 4 === s) {
                            if ("GA94" === String.fromCharCode(e[t + 3], e[t + 4], e[t + 5], e[t + 6])) {
                                i.payloadType = s,
                                i.payloadSize = n,
                                i.payload = e.subarray(t, t + n);
                                break
                            }
                            i.payload = void 0
                        }
                        t += n,
                        s = 0,
                        n = 0
                    }
                    return i
                },
                parseUserData: function(e) {
                    return 181 !== e.payload[0] || 49 !== (e.payload[1] << 8 | e.payload[2]) || "GA94" !== String.fromCharCode(e.payload[3], e.payload[4], e.payload[5], e.payload[6]) || 3 !== e.payload[7] ? null : e.payload.subarray(8, e.payload.length - 1)
                },
                parseCaptionPackets: function(e, t) {
                    var i, s, n, r, a = [];
                    if (!(64 & t[0]))
                        return a;
                    for (s = 31 & t[0],
                    i = 0; i < s; i++)
                        r = {
                            type: 3 & t[(n = 3 * i) + 2],
                            pts: e
                        },
                        4 & t[n + 2] && (r.ccData = t[n + 3] << 8 | t[n + 4],
                        a.push(r));
                    return a
                },
                discardEmulationPreventionBytes: function(e) {
                    for (var t, i, s = e.byteLength, n = [], r = 1; r < s - 2; )
                        0 === e[r] && 0 === e[r + 1] && 3 === e[r + 2] ? (n.push(r + 2),
                        r += 2) : r++;
                    if (0 === n.length)
                        return e;
                    t = s - n.length,
                    i = new Uint8Array(t);
                    var a = 0;
                    for (r = 0; r < t; a++,
                    r++)
                        a === n[0] && (a++,
                        n.shift()),
                        i[r] = e[a];
                    return i
                },
                USER_DATA_REGISTERED_ITU_T_T35: 4
            }
              , ue = F
              , ce = de
              , pe = function(e) {
                e = e || {},
                pe.prototype.init.call(this),
                this.parse708captions_ = "boolean" !== typeof e.parse708captions || e.parse708captions,
                this.captionPackets_ = [],
                this.ccStreams_ = [new ke(0,0), new ke(0,1), new ke(1,0), new ke(1,1)],
                this.parse708captions_ && (this.cc708Stream_ = new ye({
                    captionServices: e.captionServices
                })),
                this.reset(),
                this.ccStreams_.forEach((function(e) {
                    e.on("data", this.trigger.bind(this, "data")),
                    e.on("partialdone", this.trigger.bind(this, "partialdone")),
                    e.on("done", this.trigger.bind(this, "done"))
                }
                ), this),
                this.parse708captions_ && (this.cc708Stream_.on("data", this.trigger.bind(this, "data")),
                this.cc708Stream_.on("partialdone", this.trigger.bind(this, "partialdone")),
                this.cc708Stream_.on("done", this.trigger.bind(this, "done")))
            };
            pe.prototype = new ue,
            pe.prototype.push = function(e) {
                var t, i, s;
                if ("sei_rbsp" === e.nalUnitType && (t = ce.parseSei(e.escapedRBSP)).payload && t.payloadType === ce.USER_DATA_REGISTERED_ITU_T_T35 && (i = ce.parseUserData(t)))
                    if (e.dts < this.latestDts_)
                        this.ignoreNextEqualDts_ = !0;
                    else {
                        if (e.dts === this.latestDts_ && this.ignoreNextEqualDts_)
                            return this.numSameDts_--,
                            void (this.numSameDts_ || (this.ignoreNextEqualDts_ = !1));
                        s = ce.parseCaptionPackets(e.pts, i),
                        this.captionPackets_ = this.captionPackets_.concat(s),
                        this.latestDts_ !== e.dts && (this.numSameDts_ = 0),
                        this.numSameDts_++,
                        this.latestDts_ = e.dts
                    }
            }
            ,
            pe.prototype.flushCCStreams = function(e) {
                this.ccStreams_.forEach((function(t) {
                    return "flush" === e ? t.flush() : t.partialFlush()
                }
                ), this)
            }
            ,
            pe.prototype.flushStream = function(e) {
                this.captionPackets_.length ? (this.captionPackets_.forEach((function(e, t) {
                    e.presortIndex = t
                }
                )),
                this.captionPackets_.sort((function(e, t) {
                    return e.pts === t.pts ? e.presortIndex - t.presortIndex : e.pts - t.pts
                }
                )),
                this.captionPackets_.forEach((function(e) {
                    e.type < 2 ? this.dispatchCea608Packet(e) : this.dispatchCea708Packet(e)
                }
                ), this),
                this.captionPackets_.length = 0,
                this.flushCCStreams(e)) : this.flushCCStreams(e)
            }
            ,
            pe.prototype.flush = function() {
                return this.flushStream("flush")
            }
            ,
            pe.prototype.partialFlush = function() {
                return this.flushStream("partialFlush")
            }
            ,
            pe.prototype.reset = function() {
                this.latestDts_ = null,
                this.ignoreNextEqualDts_ = !1,
                this.numSameDts_ = 0,
                this.activeCea608Channel_ = [null, null],
                this.ccStreams_.forEach((function(e) {
                    e.reset()
                }
                ))
            }
            ,
            pe.prototype.dispatchCea608Packet = function(e) {
                this.setsTextOrXDSActive(e) ? this.activeCea608Channel_[e.type] = null : this.setsChannel1Active(e) ? this.activeCea608Channel_[e.type] = 0 : this.setsChannel2Active(e) && (this.activeCea608Channel_[e.type] = 1),
                null !== this.activeCea608Channel_[e.type] && this.ccStreams_[(e.type << 1) + this.activeCea608Channel_[e.type]].push(e)
            }
            ,
            pe.prototype.setsChannel1Active = function(e) {
                return 4096 === (30720 & e.ccData)
            }
            ,
            pe.prototype.setsChannel2Active = function(e) {
                return 6144 === (30720 & e.ccData)
            }
            ,
            pe.prototype.setsTextOrXDSActive = function(e) {
                return 256 === (28928 & e.ccData) || 4138 === (30974 & e.ccData) || 6186 === (30974 & e.ccData)
            }
            ,
            pe.prototype.dispatchCea708Packet = function(e) {
                this.parse708captions_ && this.cc708Stream_.push(e)
            }
            ;
            var me = {
                127: 9834,
                4128: 32,
                4129: 160,
                4133: 8230,
                4138: 352,
                4140: 338,
                4144: 9608,
                4145: 8216,
                4146: 8217,
                4147: 8220,
                4148: 8221,
                4149: 8226,
                4153: 8482,
                4154: 353,
                4156: 339,
                4157: 8480,
                4159: 376,
                4214: 8539,
                4215: 8540,
                4216: 8541,
                4217: 8542,
                4218: 9168,
                4219: 9124,
                4220: 9123,
                4221: 9135,
                4222: 9126,
                4223: 9121,
                4256: 12600
            }
              , ge = function(e) {
                return 32 <= e && e <= 127 || 160 <= e && e <= 255
            }
              , fe = function(e) {
                this.windowNum = e,
                this.reset()
            };
            fe.prototype.reset = function() {
                this.clearText(),
                this.pendingNewLine = !1,
                this.winAttr = {},
                this.penAttr = {},
                this.penLoc = {},
                this.penColor = {},
                this.visible = 0,
                this.rowLock = 0,
                this.columnLock = 0,
                this.priority = 0,
                this.relativePositioning = 0,
                this.anchorVertical = 0,
                this.anchorHorizontal = 0,
                this.anchorPoint = 0,
                this.rowCount = 1,
                this.virtualRowCount = this.rowCount + 1,
                this.columnCount = 41,
                this.windowStyle = 0,
                this.penStyle = 0
            }
            ,
            fe.prototype.getText = function() {
                return this.rows.join("\n")
            }
            ,
            fe.prototype.clearText = function() {
                this.rows = [""],
                this.rowIdx = 0
            }
            ,
            fe.prototype.newLine = function(e) {
                for (this.rows.length >= this.virtualRowCount && "function" === typeof this.beforeRowOverflow && this.beforeRowOverflow(e),
                this.rows.length > 0 && (this.rows.push(""),
                this.rowIdx++); this.rows.length > this.virtualRowCount; )
                    this.rows.shift(),
                    this.rowIdx--
            }
            ,
            fe.prototype.isEmpty = function() {
                return 0 === this.rows.length || 1 === this.rows.length && "" === this.rows[0]
            }
            ,
            fe.prototype.addText = function(e) {
                this.rows[this.rowIdx] += e
            }
            ,
            fe.prototype.backspace = function() {
                if (!this.isEmpty()) {
                    var e = this.rows[this.rowIdx];
                    this.rows[this.rowIdx] = e.substr(0, e.length - 1)
                }
            }
            ;
            var _e = function(e, t, i) {
                this.serviceNum = e,
                this.text = "",
                this.currentWindow = new fe(-1),
                this.windows = [],
                this.stream = i,
                "string" === typeof t && this.createTextDecoder(t)
            };
            _e.prototype.init = function(e, t) {
                this.startPts = e;
                for (var i = 0; i < 8; i++)
                    this.windows[i] = new fe(i),
                    "function" === typeof t && (this.windows[i].beforeRowOverflow = t)
            }
            ,
            _e.prototype.setCurrentWindow = function(e) {
                this.currentWindow = this.windows[e]
            }
            ,
            _e.prototype.createTextDecoder = function(e) {
                if ("undefined" === typeof TextDecoder)
                    this.stream.trigger("log", {
                        level: "warn",
                        message: "The `encoding` option is unsupported without TextDecoder support"
                    });
                else
                    try {
                        this.textDecoder_ = new TextDecoder(e)
                    } catch (t) {
                        this.stream.trigger("log", {
                            level: "warn",
                            message: "TextDecoder could not be created with " + e + " encoding. " + t
                        })
                    }
            }
            ;
            var ye = function(e) {
                e = e || {},
                ye.prototype.init.call(this);
                var t, i = this, s = e.captionServices || {}, n = {};
                Object.keys(s).forEach((e=>{
                    t = s[e],
                    /^SERVICE/.test(e) && (n[e] = t.encoding)
                }
                )),
                this.serviceEncodings = n,
                this.current708Packet = null,
                this.services = {},
                this.push = function(e) {
                    3 === e.type ? (i.new708Packet(),
                    i.add708Bytes(e)) : (null === i.current708Packet && i.new708Packet(),
                    i.add708Bytes(e))
                }
            };
            ye.prototype = new ue,
            ye.prototype.new708Packet = function() {
                null !== this.current708Packet && this.push708Packet(),
                this.current708Packet = {
                    data: [],
                    ptsVals: []
                }
            }
            ,
            ye.prototype.add708Bytes = function(e) {
                var t = e.ccData
                  , i = t >>> 8
                  , s = 255 & t;
                this.current708Packet.ptsVals.push(e.pts),
                this.current708Packet.data.push(i),
                this.current708Packet.data.push(s)
            }
            ,
            ye.prototype.push708Packet = function() {
                var e = this.current708Packet
                  , t = e.data
                  , i = null
                  , s = null
                  , n = 0
                  , r = t[n++];
                for (e.seq = r >> 6,
                e.sizeCode = 63 & r; n < t.length; n++)
                    s = 31 & (r = t[n++]),
                    7 === (i = r >> 5) && s > 0 && (i = r = t[n++]),
                    this.pushServiceBlock(i, n, s),
                    s > 0 && (n += s - 1)
            }
            ,
            ye.prototype.pushServiceBlock = function(e, t, i) {
                var s, n = t, r = this.current708Packet.data, a = this.services[e];
                for (a || (a = this.initService(e, n)); n < t + i && n < r.length; n++)
                    s = r[n],
                    ge(s) ? n = this.handleText(n, a) : 24 === s ? n = this.multiByteCharacter(n, a) : 16 === s ? n = this.extendedCommands(n, a) : 128 <= s && s <= 135 ? n = this.setCurrentWindow(n, a) : 152 <= s && s <= 159 ? n = this.defineWindow(n, a) : 136 === s ? n = this.clearWindows(n, a) : 140 === s ? n = this.deleteWindows(n, a) : 137 === s ? n = this.displayWindows(n, a) : 138 === s ? n = this.hideWindows(n, a) : 139 === s ? n = this.toggleWindows(n, a) : 151 === s ? n = this.setWindowAttributes(n, a) : 144 === s ? n = this.setPenAttributes(n, a) : 145 === s ? n = this.setPenColor(n, a) : 146 === s ? n = this.setPenLocation(n, a) : 143 === s ? a = this.reset(n, a) : 8 === s ? a.currentWindow.backspace() : 12 === s ? a.currentWindow.clearText() : 13 === s ? a.currentWindow.pendingNewLine = !0 : 14 === s ? a.currentWindow.clearText() : 141 === s && n++
            }
            ,
            ye.prototype.extendedCommands = function(e, t) {
                var i = this.current708Packet.data[++e];
                return ge(i) && (e = this.handleText(e, t, {
                    isExtended: !0
                })),
                e
            }
            ,
            ye.prototype.getPts = function(e) {
                return this.current708Packet.ptsVals[Math.floor(e / 2)]
            }
            ,
            ye.prototype.initService = function(e, t) {
                var i, s, n = this;
                return (i = "SERVICE" + e)in this.serviceEncodings && (s = this.serviceEncodings[i]),
                this.services[e] = new _e(e,s,n),
                this.services[e].init(this.getPts(t), (function(t) {
                    n.flushDisplayed(t, n.services[e])
                }
                )),
                this.services[e]
            }
            ,
            ye.prototype.handleText = function(e, t, i) {
                var s, n, r = i && i.isExtended, a = i && i.isMultiByte, o = this.current708Packet.data, l = r ? 4096 : 0, h = o[e], d = o[e + 1], u = t.currentWindow;
                if (a ? (n = [h, d],
                e++) : n = [h],
                t.textDecoder_ && !r)
                    s = t.textDecoder_.decode(new Uint8Array(n));
                else if (a) {
                    const e = n.map((e=>("0" + (255 & e).toString(16)).slice(-2))).join("");
                    s = String.fromCharCode(parseInt(e, 16))
                } else
                    s = function(e) {
                        var t = me[e] || e;
                        return 4096 & e && e === t ? "" : String.fromCharCode(t)
                    }(l | h);
                return u.pendingNewLine && !u.isEmpty() && u.newLine(this.getPts(e)),
                u.pendingNewLine = !1,
                u.addText(s),
                e
            }
            ,
            ye.prototype.multiByteCharacter = function(e, t) {
                var i = this.current708Packet.data
                  , s = i[e + 1]
                  , n = i[e + 2];
                return ge(s) && ge(n) && (e = this.handleText(++e, t, {
                    isMultiByte: !0
                })),
                e
            }
            ,
            ye.prototype.setCurrentWindow = function(e, t) {
                var i = 7 & this.current708Packet.data[e];
                return t.setCurrentWindow(i),
                e
            }
            ,
            ye.prototype.defineWindow = function(e, t) {
                var i = this.current708Packet.data
                  , s = i[e]
                  , n = 7 & s;
                t.setCurrentWindow(n);
                var r = t.currentWindow;
                return s = i[++e],
                r.visible = (32 & s) >> 5,
                r.rowLock = (16 & s) >> 4,
                r.columnLock = (8 & s) >> 3,
                r.priority = 7 & s,
                s = i[++e],
                r.relativePositioning = (128 & s) >> 7,
                r.anchorVertical = 127 & s,
                s = i[++e],
                r.anchorHorizontal = s,
                s = i[++e],
                r.anchorPoint = (240 & s) >> 4,
                r.rowCount = 15 & s,
                s = i[++e],
                r.columnCount = 63 & s,
                s = i[++e],
                r.windowStyle = (56 & s) >> 3,
                r.penStyle = 7 & s,
                r.virtualRowCount = r.rowCount + 1,
                e
            }
            ,
            ye.prototype.setWindowAttributes = function(e, t) {
                var i = this.current708Packet.data
                  , s = i[e]
                  , n = t.currentWindow.winAttr;
                return s = i[++e],
                n.fillOpacity = (192 & s) >> 6,
                n.fillRed = (48 & s) >> 4,
                n.fillGreen = (12 & s) >> 2,
                n.fillBlue = 3 & s,
                s = i[++e],
                n.borderType = (192 & s) >> 6,
                n.borderRed = (48 & s) >> 4,
                n.borderGreen = (12 & s) >> 2,
                n.borderBlue = 3 & s,
                s = i[++e],
                n.borderType += (128 & s) >> 5,
                n.wordWrap = (64 & s) >> 6,
                n.printDirection = (48 & s) >> 4,
                n.scrollDirection = (12 & s) >> 2,
                n.justify = 3 & s,
                s = i[++e],
                n.effectSpeed = (240 & s) >> 4,
                n.effectDirection = (12 & s) >> 2,
                n.displayEffect = 3 & s,
                e
            }
            ,
            ye.prototype.flushDisplayed = function(e, t) {
                for (var i = [], s = 0; s < 8; s++)
                    t.windows[s].visible && !t.windows[s].isEmpty() && i.push(t.windows[s].getText());
                t.endPts = e,
                t.text = i.join("\n\n"),
                this.pushCaption(t),
                t.startPts = e
            }
            ,
            ye.prototype.pushCaption = function(e) {
                "" !== e.text && (this.trigger("data", {
                    startPts: e.startPts,
                    endPts: e.endPts,
                    text: e.text,
                    stream: "cc708_" + e.serviceNum
                }),
                e.text = "",
                e.startPts = e.endPts)
            }
            ,
            ye.prototype.displayWindows = function(e, t) {
                var i = this.current708Packet.data[++e]
                  , s = this.getPts(e);
                this.flushDisplayed(s, t);
                for (var n = 0; n < 8; n++)
                    i & 1 << n && (t.windows[n].visible = 1);
                return e
            }
            ,
            ye.prototype.hideWindows = function(e, t) {
                var i = this.current708Packet.data[++e]
                  , s = this.getPts(e);
                this.flushDisplayed(s, t);
                for (var n = 0; n < 8; n++)
                    i & 1 << n && (t.windows[n].visible = 0);
                return e
            }
            ,
            ye.prototype.toggleWindows = function(e, t) {
                var i = this.current708Packet.data[++e]
                  , s = this.getPts(e);
                this.flushDisplayed(s, t);
                for (var n = 0; n < 8; n++)
                    i & 1 << n && (t.windows[n].visible ^= 1);
                return e
            }
            ,
            ye.prototype.clearWindows = function(e, t) {
                var i = this.current708Packet.data[++e]
                  , s = this.getPts(e);
                this.flushDisplayed(s, t);
                for (var n = 0; n < 8; n++)
                    i & 1 << n && t.windows[n].clearText();
                return e
            }
            ,
            ye.prototype.deleteWindows = function(e, t) {
                var i = this.current708Packet.data[++e]
                  , s = this.getPts(e);
                this.flushDisplayed(s, t);
                for (var n = 0; n < 8; n++)
                    i & 1 << n && t.windows[n].reset();
                return e
            }
            ,
            ye.prototype.setPenAttributes = function(e, t) {
                var i = this.current708Packet.data
                  , s = i[e]
                  , n = t.currentWindow.penAttr;
                return s = i[++e],
                n.textTag = (240 & s) >> 4,
                n.offset = (12 & s) >> 2,
                n.penSize = 3 & s,
                s = i[++e],
                n.italics = (128 & s) >> 7,
                n.underline = (64 & s) >> 6,
                n.edgeType = (56 & s) >> 3,
                n.fontStyle = 7 & s,
                e
            }
            ,
            ye.prototype.setPenColor = function(e, t) {
                var i = this.current708Packet.data
                  , s = i[e]
                  , n = t.currentWindow.penColor;
                return s = i[++e],
                n.fgOpacity = (192 & s) >> 6,
                n.fgRed = (48 & s) >> 4,
                n.fgGreen = (12 & s) >> 2,
                n.fgBlue = 3 & s,
                s = i[++e],
                n.bgOpacity = (192 & s) >> 6,
                n.bgRed = (48 & s) >> 4,
                n.bgGreen = (12 & s) >> 2,
                n.bgBlue = 3 & s,
                s = i[++e],
                n.edgeRed = (48 & s) >> 4,
                n.edgeGreen = (12 & s) >> 2,
                n.edgeBlue = 3 & s,
                e
            }
            ,
            ye.prototype.setPenLocation = function(e, t) {
                var i = this.current708Packet.data
                  , s = i[e]
                  , n = t.currentWindow.penLoc;
                return t.currentWindow.pendingNewLine = !0,
                s = i[++e],
                n.row = 15 & s,
                s = i[++e],
                n.column = 63 & s,
                e
            }
            ,
            ye.prototype.reset = function(e, t) {
                var i = this.getPts(e);
                return this.flushDisplayed(i, t),
                this.initService(t.serviceNum, e)
            }
            ;
            var ve = {
                42: 225,
                92: 233,
                94: 237,
                95: 243,
                96: 250,
                123: 231,
                124: 247,
                125: 209,
                126: 241,
                127: 9608,
                304: 174,
                305: 176,
                306: 189,
                307: 191,
                308: 8482,
                309: 162,
                310: 163,
                311: 9834,
                312: 224,
                313: 160,
                314: 232,
                315: 226,
                316: 234,
                317: 238,
                318: 244,
                319: 251,
                544: 193,
                545: 201,
                546: 211,
                547: 218,
                548: 220,
                549: 252,
                550: 8216,
                551: 161,
                552: 42,
                553: 39,
                554: 8212,
                555: 169,
                556: 8480,
                557: 8226,
                558: 8220,
                559: 8221,
                560: 192,
                561: 194,
                562: 199,
                563: 200,
                564: 202,
                565: 203,
                566: 235,
                567: 206,
                568: 207,
                569: 239,
                570: 212,
                571: 217,
                572: 249,
                573: 219,
                574: 171,
                575: 187,
                800: 195,
                801: 227,
                802: 205,
                803: 204,
                804: 236,
                805: 210,
                806: 242,
                807: 213,
                808: 245,
                809: 123,
                810: 125,
                811: 92,
                812: 94,
                813: 95,
                814: 124,
                815: 126,
                816: 196,
                817: 228,
                818: 214,
                819: 246,
                820: 223,
                821: 165,
                822: 164,
                823: 9474,
                824: 197,
                825: 229,
                826: 216,
                827: 248,
                828: 9484,
                829: 9488,
                830: 9492,
                831: 9496
            }
              , Te = function(e) {
                return null === e ? "" : (e = ve[e] || e,
                String.fromCharCode(e))
            }
              , be = [4352, 4384, 4608, 4640, 5376, 5408, 5632, 5664, 5888, 5920, 4096, 4864, 4896, 5120, 5152]
              , Se = function() {
                for (var e = [], t = 15; t--; )
                    e.push({
                        text: "",
                        indent: 0,
                        offset: 0
                    });
                return e
            }
              , ke = function(e, t) {
                ke.prototype.init.call(this),
                this.field_ = e || 0,
                this.dataChannel_ = t || 0,
                this.name_ = "CC" + (1 + (this.field_ << 1 | this.dataChannel_)),
                this.setConstants(),
                this.reset(),
                this.push = function(e) {
                    var t, i, s, n, r;
                    if ((t = 32639 & e.ccData) !== this.lastControlCode_) {
                        if (4096 === (61440 & t) ? this.lastControlCode_ = t : t !== this.PADDING_ && (this.lastControlCode_ = null),
                        s = t >>> 8,
                        n = 255 & t,
                        t !== this.PADDING_)
                            if (t === this.RESUME_CAPTION_LOADING_)
                                this.mode_ = "popOn";
                            else if (t === this.END_OF_CAPTION_)
                                this.mode_ = "popOn",
                                this.clearFormatting(e.pts),
                                this.flushDisplayed(e.pts),
                                i = this.displayed_,
                                this.displayed_ = this.nonDisplayed_,
                                this.nonDisplayed_ = i,
                                this.startPts_ = e.pts;
                            else if (t === this.ROLL_UP_2_ROWS_)
                                this.rollUpRows_ = 2,
                                this.setRollUp(e.pts);
                            else if (t === this.ROLL_UP_3_ROWS_)
                                this.rollUpRows_ = 3,
                                this.setRollUp(e.pts);
                            else if (t === this.ROLL_UP_4_ROWS_)
                                this.rollUpRows_ = 4,
                                this.setRollUp(e.pts);
                            else if (t === this.CARRIAGE_RETURN_)
                                this.clearFormatting(e.pts),
                                this.flushDisplayed(e.pts),
                                this.shiftRowsUp_(),
                                this.startPts_ = e.pts;
                            else if (t === this.BACKSPACE_)
                                "popOn" === this.mode_ ? this.nonDisplayed_[this.row_].text = this.nonDisplayed_[this.row_].text.slice(0, -1) : this.displayed_[this.row_].text = this.displayed_[this.row_].text.slice(0, -1);
                            else if (t === this.ERASE_DISPLAYED_MEMORY_)
                                this.flushDisplayed(e.pts),
                                this.displayed_ = Se();
                            else if (t === this.ERASE_NON_DISPLAYED_MEMORY_)
                                this.nonDisplayed_ = Se();
                            else if (t === this.RESUME_DIRECT_CAPTIONING_)
                                "paintOn" !== this.mode_ && (this.flushDisplayed(e.pts),
                                this.displayed_ = Se()),
                                this.mode_ = "paintOn",
                                this.startPts_ = e.pts;
                            else if (this.isSpecialCharacter(s, n))
                                r = Te((s = (3 & s) << 8) | n),
                                this[this.mode_](e.pts, r),
                                this.column_++;
                            else if (this.isExtCharacter(s, n))
                                "popOn" === this.mode_ ? this.nonDisplayed_[this.row_].text = this.nonDisplayed_[this.row_].text.slice(0, -1) : this.displayed_[this.row_].text = this.displayed_[this.row_].text.slice(0, -1),
                                r = Te((s = (3 & s) << 8) | n),
                                this[this.mode_](e.pts, r),
                                this.column_++;
                            else if (this.isMidRowCode(s, n))
                                this.clearFormatting(e.pts),
                                this[this.mode_](e.pts, " "),
                                this.column_++,
                                14 === (14 & n) && this.addFormatting(e.pts, ["i"]),
                                1 === (1 & n) && this.addFormatting(e.pts, ["u"]);
                            else if (this.isOffsetControlCode(s, n)) {
                                const e = 3 & n;
                                this.nonDisplayed_[this.row_].offset = e,
                                this.column_ += e
                            } else if (this.isPAC(s, n)) {
                                var a = be.indexOf(7968 & t);
                                if ("rollUp" === this.mode_ && (a - this.rollUpRows_ + 1 < 0 && (a = this.rollUpRows_ - 1),
                                this.setRollUp(e.pts, a)),
                                a !== this.row_ && (this.clearFormatting(e.pts),
                                this.row_ = a),
                                1 & n && -1 === this.formatting_.indexOf("u") && this.addFormatting(e.pts, ["u"]),
                                16 === (16 & t)) {
                                    const e = (14 & t) >> 1;
                                    this.column_ = 4 * e,
                                    this.nonDisplayed_[this.row_].indent += e
                                }
                                this.isColorPAC(n) && 14 === (14 & n) && this.addFormatting(e.pts, ["i"])
                            } else
                                this.isNormalChar(s) && (0 === n && (n = null),
                                r = Te(s),
                                r += Te(n),
                                this[this.mode_](e.pts, r),
                                this.column_ += r.length)
                    } else
                        this.lastControlCode_ = null
                }
            };
            ke.prototype = new ue,
            ke.prototype.flushDisplayed = function(e) {
                const t = e=>{
                    this.trigger("log", {
                        level: "warn",
                        message: "Skipping a malformed 608 caption at index " + e + "."
                    })
                }
                  , i = [];
                this.displayed_.forEach(((e,s)=>{
                    if (e && e.text && e.text.length) {
                        try {
                            e.text = e.text.trim()
                        } catch (n) {
                            t(s)
                        }
                        e.text.length && i.push({
                            text: e.text,
                            line: s + 1,
                            position: 10 + Math.min(70, 10 * e.indent) + 2.5 * e.offset
                        })
                    } else
                        void 0 !== e && null !== e || t(s)
                }
                )),
                i.length && this.trigger("data", {
                    startPts: this.startPts_,
                    endPts: e,
                    content: i,
                    stream: this.name_
                })
            }
            ,
            ke.prototype.reset = function() {
                this.mode_ = "popOn",
                this.topRow_ = 0,
                this.startPts_ = 0,
                this.displayed_ = Se(),
                this.nonDisplayed_ = Se(),
                this.lastControlCode_ = null,
                this.column_ = 0,
                this.row_ = 14,
                this.rollUpRows_ = 2,
                this.formatting_ = []
            }
            ,
            ke.prototype.setConstants = function() {
                0 === this.dataChannel_ ? (this.BASE_ = 16,
                this.EXT_ = 17,
                this.CONTROL_ = (20 | this.field_) << 8,
                this.OFFSET_ = 23) : 1 === this.dataChannel_ && (this.BASE_ = 24,
                this.EXT_ = 25,
                this.CONTROL_ = (28 | this.field_) << 8,
                this.OFFSET_ = 31),
                this.PADDING_ = 0,
                this.RESUME_CAPTION_LOADING_ = 32 | this.CONTROL_,
                this.END_OF_CAPTION_ = 47 | this.CONTROL_,
                this.ROLL_UP_2_ROWS_ = 37 | this.CONTROL_,
                this.ROLL_UP_3_ROWS_ = 38 | this.CONTROL_,
                this.ROLL_UP_4_ROWS_ = 39 | this.CONTROL_,
                this.CARRIAGE_RETURN_ = 45 | this.CONTROL_,
                this.RESUME_DIRECT_CAPTIONING_ = 41 | this.CONTROL_,
                this.BACKSPACE_ = 33 | this.CONTROL_,
                this.ERASE_DISPLAYED_MEMORY_ = 44 | this.CONTROL_,
                this.ERASE_NON_DISPLAYED_MEMORY_ = 46 | this.CONTROL_
            }
            ,
            ke.prototype.isSpecialCharacter = function(e, t) {
                return e === this.EXT_ && t >= 48 && t <= 63
            }
            ,
            ke.prototype.isExtCharacter = function(e, t) {
                return (e === this.EXT_ + 1 || e === this.EXT_ + 2) && t >= 32 && t <= 63
            }
            ,
            ke.prototype.isMidRowCode = function(e, t) {
                return e === this.EXT_ && t >= 32 && t <= 47
            }
            ,
            ke.prototype.isOffsetControlCode = function(e, t) {
                return e === this.OFFSET_ && t >= 33 && t <= 35
            }
            ,
            ke.prototype.isPAC = function(e, t) {
                return e >= this.BASE_ && e < this.BASE_ + 8 && t >= 64 && t <= 127
            }
            ,
            ke.prototype.isColorPAC = function(e) {
                return e >= 64 && e <= 79 || e >= 96 && e <= 127
            }
            ,
            ke.prototype.isNormalChar = function(e) {
                return e >= 32 && e <= 127
            }
            ,
            ke.prototype.setRollUp = function(e, t) {
                if ("rollUp" !== this.mode_ && (this.row_ = 14,
                this.mode_ = "rollUp",
                this.flushDisplayed(e),
                this.nonDisplayed_ = Se(),
                this.displayed_ = Se()),
                void 0 !== t && t !== this.row_)
                    for (var i = 0; i < this.rollUpRows_; i++)
                        this.displayed_[t - i] = this.displayed_[this.row_ - i],
                        this.displayed_[this.row_ - i] = {
                            text: "",
                            indent: 0,
                            offset: 0
                        };
                void 0 === t && (t = this.row_),
                this.topRow_ = t - this.rollUpRows_ + 1
            }
            ,
            ke.prototype.addFormatting = function(e, t) {
                this.formatting_ = this.formatting_.concat(t);
                var i = t.reduce((function(e, t) {
                    return e + "<" + t + ">"
                }
                ), "");
                this[this.mode_](e, i)
            }
            ,
            ke.prototype.clearFormatting = function(e) {
                if (this.formatting_.length) {
                    var t = this.formatting_.reverse().reduce((function(e, t) {
                        return e + "</" + t + ">"
                    }
                    ), "");
                    this.formatting_ = [],
                    this[this.mode_](e, t)
                }
            }
            ,
            ke.prototype.popOn = function(e, t) {
                var i = this.nonDisplayed_[this.row_].text;
                i += t,
                this.nonDisplayed_[this.row_].text = i
            }
            ,
            ke.prototype.rollUp = function(e, t) {
                var i = this.displayed_[this.row_].text;
                i += t,
                this.displayed_[this.row_].text = i
            }
            ,
            ke.prototype.shiftRowsUp_ = function() {
                var e;
                for (e = 0; e < this.topRow_; e++)
                    this.displayed_[e] = {
                        text: "",
                        indent: 0,
                        offset: 0
                    };
                for (e = this.row_ + 1; e < 15; e++)
                    this.displayed_[e] = {
                        text: "",
                        indent: 0,
                        offset: 0
                    };
                for (e = this.topRow_; e < this.row_; e++)
                    this.displayed_[e] = this.displayed_[e + 1];
                this.displayed_[this.row_] = {
                    text: "",
                    indent: 0,
                    offset: 0
                }
            }
            ,
            ke.prototype.paintOn = function(e, t) {
                var i = this.displayed_[this.row_].text;
                i += t,
                this.displayed_[this.row_].text = i
            }
            ;
            var Ce = {
                CaptionStream: pe,
                Cea608Stream: ke,
                Cea708Stream: ye
            }
              , Ee = {
                H264_STREAM_TYPE: 27,
                ADTS_STREAM_TYPE: 15,
                METADATA_STREAM_TYPE: 21
            }
              , we = F
              , xe = "shared"
              , Ie = function(e, t) {
                var i = 1;
                for (e > t && (i = -1); Math.abs(t - e) > 4294967296; )
                    e += 8589934592 * i;
                return e
            }
              , Pe = function(e) {
                var t, i;
                Pe.prototype.init.call(this),
                this.type_ = e || xe,
                this.push = function(e) {
                    "metadata" !== e.type ? this.type_ !== xe && e.type !== this.type_ || (void 0 === i && (i = e.dts),
                    e.dts = Ie(e.dts, i),
                    e.pts = Ie(e.pts, i),
                    t = e.dts,
                    this.trigger("data", e)) : this.trigger("data", e)
                }
                ,
                this.flush = function() {
                    i = t,
                    this.trigger("done")
                }
                ,
                this.endTimeline = function() {
                    this.flush(),
                    this.trigger("endedtimeline")
                }
                ,
                this.discontinuity = function() {
                    i = void 0,
                    t = void 0
                }
                ,
                this.reset = function() {
                    this.discontinuity(),
                    this.trigger("reset")
                }
            };
            Pe.prototype = new we;
            var Ae, Le = {
                TimestampRolloverStream: Pe,
                handleRollover: Ie
            }, De = (e,t,i)=>{
                if (!e)
                    return -1;
                for (var s = i; s < e.length; s++)
                    if (e[s] === t)
                        return s;
                return -1
            }
            , Oe = 3, Me = function(e, t, i) {
                var s, n = "";
                for (s = t; s < i; s++)
                    n += "%" + ("00" + e[s].toString(16)).slice(-2);
                return n
            }, Re = function(e, t, i) {
                return decodeURIComponent(Me(e, t, i))
            }, Ue = function(e, t, i) {
                return unescape(Me(e, t, i))
            }, Be = function(e) {
                return e[0] << 21 | e[1] << 14 | e[2] << 7 | e[3]
            }, Ne = {
                APIC: function(e) {
                    var t, i, s = 1;
                    e.data[0] === Oe && ((t = De(e.data, 0, s)) < 0 || (e.mimeType = Ue(e.data, s, t),
                    s = t + 1,
                    e.pictureType = e.data[s],
                    s++,
                    (i = De(e.data, 0, s)) < 0 || (e.description = Re(e.data, s, i),
                    s = i + 1,
                    "--\x3e" === e.mimeType ? e.url = Ue(e.data, s, e.data.length) : e.pictureData = e.data.subarray(s, e.data.length))))
                },
                "T*": function(e) {
                    e.data[0] === Oe && (e.value = Re(e.data, 1, e.data.length).replace(/\0*$/, ""),
                    e.values = e.value.split("\0"))
                },
                TXXX: function(e) {
                    var t;
                    e.data[0] === Oe && -1 !== (t = De(e.data, 0, 1)) && (e.description = Re(e.data, 1, t),
                    e.value = Re(e.data, t + 1, e.data.length).replace(/\0*$/, ""),
                    e.data = e.value)
                },
                "W*": function(e) {
                    e.url = Ue(e.data, 0, e.data.length).replace(/\0.*$/, "")
                },
                WXXX: function(e) {
                    var t;
                    e.data[0] === Oe && -1 !== (t = De(e.data, 0, 1)) && (e.description = Re(e.data, 1, t),
                    e.url = Ue(e.data, t + 1, e.data.length).replace(/\0.*$/, ""))
                },
                PRIV: function(e) {
                    var t;
                    for (t = 0; t < e.data.length; t++)
                        if (0 === e.data[t]) {
                            e.owner = Ue(e.data, 0, t);
                            break
                        }
                    e.privateData = e.data.subarray(t + 1),
                    e.data = e.privateData
                }
            }, Fe = {
                parseId3Frames: function(e) {
                    var t, i = 10, s = 0, n = [];
                    if (!(e.length < 10 || e[0] !== "I".charCodeAt(0) || e[1] !== "D".charCodeAt(0) || e[2] !== "3".charCodeAt(0))) {
                        s = Be(e.subarray(6, 10)),
                        s += 10,
                        64 & e[5] && (i += 4,
                        i += Be(e.subarray(10, 14)),
                        s -= Be(e.subarray(16, 20)));
                        do {
                            if ((t = Be(e.subarray(i + 4, i + 8))) < 1)
                                break;
                            var r = {
                                id: String.fromCharCode(e[i], e[i + 1], e[i + 2], e[i + 3]),
                                data: e.subarray(i + 10, i + t + 10)
                            };
                            r.key = r.id,
                            Ne[r.id] ? Ne[r.id](r) : "T" === r.id[0] ? Ne["T*"](r) : "W" === r.id[0] && Ne["W*"](r),
                            n.push(r),
                            i += 10,
                            i += t
                        } while (i < s);
                        return n
                    }
                },
                parseSyncSafeInteger: Be,
                frameParsers: Ne
            }, je = Ee, $e = Fe;
            (Ae = function(e) {
                var t, i = {
                    descriptor: e && e.descriptor
                }, s = 0, n = [], r = 0;
                if (Ae.prototype.init.call(this),
                this.dispatchType = je.METADATA_STREAM_TYPE.toString(16),
                i.descriptor)
                    for (t = 0; t < i.descriptor.length; t++)
                        this.dispatchType += ("00" + i.descriptor[t].toString(16)).slice(-2);
                this.push = function(e) {
                    var t, i, a, o, l;
                    if ("timed-metadata" === e.type)
                        if (e.dataAlignmentIndicator && (r = 0,
                        n.length = 0),
                        0 === n.length && (e.data.length < 10 || e.data[0] !== "I".charCodeAt(0) || e.data[1] !== "D".charCodeAt(0) || e.data[2] !== "3".charCodeAt(0)))
                            this.trigger("log", {
                                level: "warn",
                                message: "Skipping unrecognized metadata packet"
                            });
                        else if (n.push(e),
                        r += e.data.byteLength,
                        1 === n.length && (s = $e.parseSyncSafeInteger(e.data.subarray(6, 10)),
                        s += 10),
                        !(r < s)) {
                            for (t = {
                                data: new Uint8Array(s),
                                frames: [],
                                pts: n[0].pts,
                                dts: n[0].dts
                            },
                            l = 0; l < s; )
                                t.data.set(n[0].data.subarray(0, s - l), l),
                                l += n[0].data.byteLength,
                                r -= n[0].data.byteLength,
                                n.shift();
                            i = 10,
                            64 & t.data[5] && (i += 4,
                            i += $e.parseSyncSafeInteger(t.data.subarray(10, 14)),
                            s -= $e.parseSyncSafeInteger(t.data.subarray(16, 20)));
                            do {
                                if ((a = $e.parseSyncSafeInteger(t.data.subarray(i + 4, i + 8))) < 1) {
                                    this.trigger("log", {
                                        level: "warn",
                                        message: "Malformed ID3 frame encountered. Skipping remaining metadata parsing."
                                    });
                                    break
                                }
                                if ((o = {
                                    id: String.fromCharCode(t.data[i], t.data[i + 1], t.data[i + 2], t.data[i + 3]),
                                    data: t.data.subarray(i + 10, i + a + 10)
                                }).key = o.id,
                                $e.frameParsers[o.id] ? $e.frameParsers[o.id](o) : "T" === o.id[0] ? $e.frameParsers["T*"](o) : "W" === o.id[0] && $e.frameParsers["W*"](o),
                                "com.apple.streaming.transportStreamTimestamp" === o.owner) {
                                    var h = o.data
                                      , d = (1 & h[3]) << 30 | h[4] << 22 | h[5] << 14 | h[6] << 6 | h[7] >>> 2;
                                    d *= 4,
                                    d += 3 & h[7],
                                    o.timeStamp = d,
                                    void 0 === t.pts && void 0 === t.dts && (t.pts = o.timeStamp,
                                    t.dts = o.timeStamp),
                                    this.trigger("timestamp", o)
                                }
                                t.frames.push(o),
                                i += 10,
                                i += a
                            } while (i < s);
                            this.trigger("data", t)
                        }
                }
            }
            ).prototype = new F;
            var qe, He, Ve, ze = Ae, We = F, Ge = Ce, Ke = Ee, Qe = Le.TimestampRolloverStream, Xe = 188;
            (qe = function() {
                var e = new Uint8Array(Xe)
                  , t = 0;
                qe.prototype.init.call(this),
                this.push = function(i) {
                    var s, n = 0, r = Xe;
                    for (t ? ((s = new Uint8Array(i.byteLength + t)).set(e.subarray(0, t)),
                    s.set(i, t),
                    t = 0) : s = i; r < s.byteLength; )
                        71 !== s[n] || 71 !== s[r] ? (n++,
                        r++) : (this.trigger("data", s.subarray(n, r)),
                        n += Xe,
                        r += Xe);
                    n < s.byteLength && (e.set(s.subarray(n), 0),
                    t = s.byteLength - n)
                }
                ,
                this.flush = function() {
                    t === Xe && 71 === e[0] && (this.trigger("data", e),
                    t = 0),
                    this.trigger("done")
                }
                ,
                this.endTimeline = function() {
                    this.flush(),
                    this.trigger("endedtimeline")
                }
                ,
                this.reset = function() {
                    t = 0,
                    this.trigger("reset")
                }
            }
            ).prototype = new We,
            (He = function() {
                var e, t, i, s;
                He.prototype.init.call(this),
                s = this,
                this.packetsWaitingForPmt = [],
                this.programMapTable = void 0,
                e = function(e, s) {
                    var n = 0;
                    s.payloadUnitStartIndicator && (n += e[n] + 1),
                    "pat" === s.type ? t(e.subarray(n), s) : i(e.subarray(n), s)
                }
                ,
                t = function(e, t) {
                    t.section_number = e[7],
                    t.last_section_number = e[8],
                    s.pmtPid = (31 & e[10]) << 8 | e[11],
                    t.pmtPid = s.pmtPid
                }
                ,
                i = function(e, t) {
                    var i, n;
                    if (1 & e[5]) {
                        for (s.programMapTable = {
                            video: null,
                            audio: null,
                            "timed-metadata": {}
                        },
                        i = 3 + ((15 & e[1]) << 8 | e[2]) - 4,
                        n = 12 + ((15 & e[10]) << 8 | e[11]); n < i; ) {
                            var r = e[n]
                              , a = (31 & e[n + 1]) << 8 | e[n + 2];
                            r === Ke.H264_STREAM_TYPE && null === s.programMapTable.video ? s.programMapTable.video = a : r === Ke.ADTS_STREAM_TYPE && null === s.programMapTable.audio ? s.programMapTable.audio = a : r === Ke.METADATA_STREAM_TYPE && (s.programMapTable["timed-metadata"][a] = r),
                            n += 5 + ((15 & e[n + 3]) << 8 | e[n + 4])
                        }
                        t.programMapTable = s.programMapTable
                    }
                }
                ,
                this.push = function(t) {
                    var i = {}
                      , s = 4;
                    if (i.payloadUnitStartIndicator = !!(64 & t[1]),
                    i.pid = 31 & t[1],
                    i.pid <<= 8,
                    i.pid |= t[2],
                    (48 & t[3]) >>> 4 > 1 && (s += t[s] + 1),
                    0 === i.pid)
                        i.type = "pat",
                        e(t.subarray(s), i),
                        this.trigger("data", i);
                    else if (i.pid === this.pmtPid)
                        for (i.type = "pmt",
                        e(t.subarray(s), i),
                        this.trigger("data", i); this.packetsWaitingForPmt.length; )
                            this.processPes_.apply(this, this.packetsWaitingForPmt.shift());
                    else
                        void 0 === this.programMapTable ? this.packetsWaitingForPmt.push([t, s, i]) : this.processPes_(t, s, i)
                }
                ,
                this.processPes_ = function(e, t, i) {
                    i.pid === this.programMapTable.video ? i.streamType = Ke.H264_STREAM_TYPE : i.pid === this.programMapTable.audio ? i.streamType = Ke.ADTS_STREAM_TYPE : i.streamType = this.programMapTable["timed-metadata"][i.pid],
                    i.type = "pes",
                    i.data = e.subarray(t),
                    this.trigger("data", i)
                }
            }
            ).prototype = new We,
            He.STREAM_TYPES = {
                h264: 27,
                adts: 15
            },
            (Ve = function() {
                var e, t = this, i = !1, s = {
                    data: [],
                    size: 0
                }, n = {
                    data: [],
                    size: 0
                }, r = {
                    data: [],
                    size: 0
                }, a = function(e, i, s) {
                    var n, r, a = new Uint8Array(e.size), o = {
                        type: i
                    }, l = 0, h = 0;
                    if (e.data.length && !(e.size < 9)) {
                        for (o.trackId = e.data[0].pid,
                        l = 0; l < e.data.length; l++)
                            r = e.data[l],
                            a.set(r.data, h),
                            h += r.data.byteLength;
                        !function(e, t) {
                            var i;
                            const s = e[0] << 16 | e[1] << 8 | e[2];
                            t.data = new Uint8Array,
                            1 === s && (t.packetLength = 6 + (e[4] << 8 | e[5]),
                            t.dataAlignmentIndicator = 0 !== (4 & e[6]),
                            192 & (i = e[7]) && (t.pts = (14 & e[9]) << 27 | (255 & e[10]) << 20 | (254 & e[11]) << 12 | (255 & e[12]) << 5 | (254 & e[13]) >>> 3,
                            t.pts *= 4,
                            t.pts += (6 & e[13]) >>> 1,
                            t.dts = t.pts,
                            64 & i && (t.dts = (14 & e[14]) << 27 | (255 & e[15]) << 20 | (254 & e[16]) << 12 | (255 & e[17]) << 5 | (254 & e[18]) >>> 3,
                            t.dts *= 4,
                            t.dts += (6 & e[18]) >>> 1)),
                            t.data = e.subarray(9 + e[8]))
                        }(a, o),
                        n = "video" === i || o.packetLength <= e.size,
                        (s || n) && (e.size = 0,
                        e.data.length = 0),
                        n && t.trigger("data", o)
                    }
                };
                Ve.prototype.init.call(this),
                this.push = function(o) {
                    ({
                        pat: function() {},
                        pes: function() {
                            var e, t;
                            switch (o.streamType) {
                            case Ke.H264_STREAM_TYPE:
                                e = s,
                                t = "video";
                                break;
                            case Ke.ADTS_STREAM_TYPE:
                                e = n,
                                t = "audio";
                                break;
                            case Ke.METADATA_STREAM_TYPE:
                                e = r,
                                t = "timed-metadata";
                                break;
                            default:
                                return
                            }
                            o.payloadUnitStartIndicator && a(e, t, !0),
                            e.data.push(o),
                            e.size += o.data.byteLength
                        },
                        pmt: function() {
                            var s = {
                                type: "metadata",
                                tracks: []
                            };
                            null !== (e = o.programMapTable).video && s.tracks.push({
                                timelineStartInfo: {
                                    baseMediaDecodeTime: 0
                                },
                                id: +e.video,
                                codec: "avc",
                                type: "video"
                            }),
                            null !== e.audio && s.tracks.push({
                                timelineStartInfo: {
                                    baseMediaDecodeTime: 0
                                },
                                id: +e.audio,
                                codec: "adts",
                                type: "audio"
                            }),
                            i = !0,
                            t.trigger("data", s)
                        }
                    })[o.type]()
                }
                ,
                this.reset = function() {
                    s.size = 0,
                    s.data.length = 0,
                    n.size = 0,
                    n.data.length = 0,
                    this.trigger("reset")
                }
                ,
                this.flushStreams_ = function() {
                    a(s, "video"),
                    a(n, "audio"),
                    a(r, "timed-metadata")
                }
                ,
                this.flush = function() {
                    if (!i && e) {
                        var s = {
                            type: "metadata",
                            tracks: []
                        };
                        null !== e.video && s.tracks.push({
                            timelineStartInfo: {
                                baseMediaDecodeTime: 0
                            },
                            id: +e.video,
                            codec: "avc",
                            type: "video"
                        }),
                        null !== e.audio && s.tracks.push({
                            timelineStartInfo: {
                                baseMediaDecodeTime: 0
                            },
                            id: +e.audio,
                            codec: "adts",
                            type: "audio"
                        }),
                        t.trigger("data", s)
                    }
                    i = !1,
                    this.flushStreams_(),
                    this.trigger("done")
                }
            }
            ).prototype = new We;
            var Ye = {
                PAT_PID: 0,
                MP2T_PACKET_LENGTH: Xe,
                TransportPacketStream: qe,
                TransportParseStream: He,
                ElementaryStream: Ve,
                TimestampRolloverStream: Qe,
                CaptionStream: Ge.CaptionStream,
                Cea608Stream: Ge.Cea608Stream,
                Cea708Stream: Ge.Cea708Stream,
                MetadataStream: ze
            };
            for (var Je in Ke)
                Ke.hasOwnProperty(Je) && (Ye[Je] = Ke[Je]);
            var Ze, et = Ye, tt = ne.ONE_SECOND_IN_TS, it = [96e3, 88200, 64e3, 48e3, 44100, 32e3, 24e3, 22050, 16e3, 12e3, 11025, 8e3, 7350];
            (Ze = function(e) {
                var t, i = 0;
                Ze.prototype.init.call(this),
                this.skipWarn_ = function(e, t) {
                    this.trigger("log", {
                        level: "warn",
                        message: `adts skiping bytes ${e} to ${t} in frame ${i} outside syncword`
                    })
                }
                ,
                this.push = function(s) {
                    var n, r, a, o, l, h = 0;
                    if (e || (i = 0),
                    "audio" === s.type) {
                        var d;
                        for (t && t.length ? (a = t,
                        (t = new Uint8Array(a.byteLength + s.data.byteLength)).set(a),
                        t.set(s.data, a.byteLength)) : t = s.data; h + 7 < t.length; )
                            if (255 === t[h] && 240 === (246 & t[h + 1])) {
                                if ("number" === typeof d && (this.skipWarn_(d, h),
                                d = null),
                                r = 2 * (1 & ~t[h + 1]),
                                n = (3 & t[h + 3]) << 11 | t[h + 4] << 3 | (224 & t[h + 5]) >> 5,
                                l = (o = 1024 * (1 + (3 & t[h + 6]))) * tt / it[(60 & t[h + 2]) >>> 2],
                                t.byteLength - h < n)
                                    break;
                                this.trigger("data", {
                                    pts: s.pts + i * l,
                                    dts: s.dts + i * l,
                                    sampleCount: o,
                                    audioobjecttype: 1 + (t[h + 2] >>> 6 & 3),
                                    channelcount: (1 & t[h + 2]) << 2 | (192 & t[h + 3]) >>> 6,
                                    samplerate: it[(60 & t[h + 2]) >>> 2],
                                    samplingfrequencyindex: (60 & t[h + 2]) >>> 2,
                                    samplesize: 16,
                                    data: t.subarray(h + 7 + r, h + n)
                                }),
                                i++,
                                h += n
                            } else
                                "number" !== typeof d && (d = h),
                                h++;
                        "number" === typeof d && (this.skipWarn_(d, h),
                        d = null),
                        t = t.subarray(h)
                    }
                }
                ,
                this.flush = function() {
                    i = 0,
                    this.trigger("done")
                }
                ,
                this.reset = function() {
                    t = void 0,
                    this.trigger("reset")
                }
                ,
                this.endTimeline = function() {
                    t = void 0,
                    this.trigger("endedtimeline")
                }
            }
            ).prototype = new F;
            var st, nt, rt, at = Ze, ot = F, lt = function(e) {
                var t = e.byteLength
                  , i = 0
                  , s = 0;
                this.length = function() {
                    return 8 * t
                }
                ,
                this.bitsAvailable = function() {
                    return 8 * t + s
                }
                ,
                this.loadWord = function() {
                    var n = e.byteLength - t
                      , r = new Uint8Array(4)
                      , a = Math.min(4, t);
                    if (0 === a)
                        throw new Error("no bytes available");
                    r.set(e.subarray(n, n + a)),
                    i = new DataView(r.buffer).getUint32(0),
                    s = 8 * a,
                    t -= a
                }
                ,
                this.skipBits = function(e) {
                    var n;
                    s > e ? (i <<= e,
                    s -= e) : (e -= s,
                    e -= 8 * (n = Math.floor(e / 8)),
                    t -= n,
                    this.loadWord(),
                    i <<= e,
                    s -= e)
                }
                ,
                this.readBits = function(e) {
                    var n = Math.min(s, e)
                      , r = i >>> 32 - n;
                    return (s -= n) > 0 ? i <<= n : t > 0 && this.loadWord(),
                    (n = e - n) > 0 ? r << n | this.readBits(n) : r
                }
                ,
                this.skipLeadingZeros = function() {
                    var e;
                    for (e = 0; e < s; ++e)
                        if (0 !== (i & 2147483648 >>> e))
                            return i <<= e,
                            s -= e,
                            e;
                    return this.loadWord(),
                    e + this.skipLeadingZeros()
                }
                ,
                this.skipUnsignedExpGolomb = function() {
                    this.skipBits(1 + this.skipLeadingZeros())
                }
                ,
                this.skipExpGolomb = function() {
                    this.skipBits(1 + this.skipLeadingZeros())
                }
                ,
                this.readUnsignedExpGolomb = function() {
                    var e = this.skipLeadingZeros();
                    return this.readBits(e + 1) - 1
                }
                ,
                this.readExpGolomb = function() {
                    var e = this.readUnsignedExpGolomb();
                    return 1 & e ? 1 + e >>> 1 : -1 * (e >>> 1)
                }
                ,
                this.readBoolean = function() {
                    return 1 === this.readBits(1)
                }
                ,
                this.readUnsignedByte = function() {
                    return this.readBits(8)
                }
                ,
                this.loadWord()
            };
            (nt = function() {
                var e, t, i = 0;
                nt.prototype.init.call(this),
                this.push = function(s) {
                    var n;
                    t ? ((n = new Uint8Array(t.byteLength + s.data.byteLength)).set(t),
                    n.set(s.data, t.byteLength),
                    t = n) : t = s.data;
                    for (var r = t.byteLength; i < r - 3; i++)
                        if (1 === t[i + 2]) {
                            e = i + 5;
                            break
                        }
                    for (; e < r; )
                        switch (t[e]) {
                        case 0:
                            if (0 !== t[e - 1]) {
                                e += 2;
                                break
                            }
                            if (0 !== t[e - 2]) {
                                e++;
                                break
                            }
                            i + 3 !== e - 2 && this.trigger("data", t.subarray(i + 3, e - 2));
                            do {
                                e++
                            } while (1 !== t[e] && e < r);
                            i = e - 2,
                            e += 3;
                            break;
                        case 1:
                            if (0 !== t[e - 1] || 0 !== t[e - 2]) {
                                e += 3;
                                break
                            }
                            this.trigger("data", t.subarray(i + 3, e - 2)),
                            i = e - 2,
                            e += 3;
                            break;
                        default:
                            e += 3
                        }
                    t = t.subarray(i),
                    e -= i,
                    i = 0
                }
                ,
                this.reset = function() {
                    t = null,
                    i = 0,
                    this.trigger("reset")
                }
                ,
                this.flush = function() {
                    t && t.byteLength > 3 && this.trigger("data", t.subarray(i + 3)),
                    t = null,
                    i = 0,
                    this.trigger("done")
                }
                ,
                this.endTimeline = function() {
                    this.flush(),
                    this.trigger("endedtimeline")
                }
            }
            ).prototype = new ot,
            rt = {
                100: !0,
                110: !0,
                122: !0,
                244: !0,
                44: !0,
                83: !0,
                86: !0,
                118: !0,
                128: !0,
                138: !0,
                139: !0,
                134: !0
            },
            (st = function() {
                var e, t, i, s, n, r, a, o = new nt;
                st.prototype.init.call(this),
                e = this,
                this.push = function(e) {
                    "video" === e.type && (t = e.trackId,
                    i = e.pts,
                    s = e.dts,
                    o.push(e))
                }
                ,
                o.on("data", (function(a) {
                    var o = {
                        trackId: t,
                        pts: i,
                        dts: s,
                        data: a,
                        nalUnitTypeCode: 31 & a[0]
                    };
                    switch (o.nalUnitTypeCode) {
                    case 5:
                        o.nalUnitType = "slice_layer_without_partitioning_rbsp_idr";
                        break;
                    case 6:
                        o.nalUnitType = "sei_rbsp",
                        o.escapedRBSP = n(a.subarray(1));
                        break;
                    case 7:
                        o.nalUnitType = "seq_parameter_set_rbsp",
                        o.escapedRBSP = n(a.subarray(1)),
                        o.config = r(o.escapedRBSP);
                        break;
                    case 8:
                        o.nalUnitType = "pic_parameter_set_rbsp";
                        break;
                    case 9:
                        o.nalUnitType = "access_unit_delimiter_rbsp"
                    }
                    e.trigger("data", o)
                }
                )),
                o.on("done", (function() {
                    e.trigger("done")
                }
                )),
                o.on("partialdone", (function() {
                    e.trigger("partialdone")
                }
                )),
                o.on("reset", (function() {
                    e.trigger("reset")
                }
                )),
                o.on("endedtimeline", (function() {
                    e.trigger("endedtimeline")
                }
                )),
                this.flush = function() {
                    o.flush()
                }
                ,
                this.partialFlush = function() {
                    o.partialFlush()
                }
                ,
                this.reset = function() {
                    o.reset()
                }
                ,
                this.endTimeline = function() {
                    o.endTimeline()
                }
                ,
                a = function(e, t) {
                    var i, s = 8, n = 8;
                    for (i = 0; i < e; i++)
                        0 !== n && (n = (s + t.readExpGolomb() + 256) % 256),
                        s = 0 === n ? s : n
                }
                ,
                n = function(e) {
                    for (var t, i, s = e.byteLength, n = [], r = 1; r < s - 2; )
                        0 === e[r] && 0 === e[r + 1] && 3 === e[r + 2] ? (n.push(r + 2),
                        r += 2) : r++;
                    if (0 === n.length)
                        return e;
                    t = s - n.length,
                    i = new Uint8Array(t);
                    var a = 0;
                    for (r = 0; r < t; a++,
                    r++)
                        a === n[0] && (a++,
                        n.shift()),
                        i[r] = e[a];
                    return i
                }
                ,
                r = function(e) {
                    var t, i, s, n, r, o, l, h, d, u, c, p, m = 0, g = 0, f = 0, _ = 0, y = [1, 1];
                    if (i = (t = new lt(e)).readUnsignedByte(),
                    n = t.readUnsignedByte(),
                    s = t.readUnsignedByte(),
                    t.skipUnsignedExpGolomb(),
                    rt[i] && (3 === (r = t.readUnsignedExpGolomb()) && t.skipBits(1),
                    t.skipUnsignedExpGolomb(),
                    t.skipUnsignedExpGolomb(),
                    t.skipBits(1),
                    t.readBoolean()))
                        for (c = 3 !== r ? 8 : 12,
                        p = 0; p < c; p++)
                            t.readBoolean() && a(p < 6 ? 16 : 64, t);
                    if (t.skipUnsignedExpGolomb(),
                    0 === (o = t.readUnsignedExpGolomb()))
                        t.readUnsignedExpGolomb();
                    else if (1 === o)
                        for (t.skipBits(1),
                        t.skipExpGolomb(),
                        t.skipExpGolomb(),
                        l = t.readUnsignedExpGolomb(),
                        p = 0; p < l; p++)
                            t.skipExpGolomb();
                    if (t.skipUnsignedExpGolomb(),
                    t.skipBits(1),
                    h = t.readUnsignedExpGolomb(),
                    d = t.readUnsignedExpGolomb(),
                    0 === (u = t.readBits(1)) && t.skipBits(1),
                    t.skipBits(1),
                    t.readBoolean() && (m = t.readUnsignedExpGolomb(),
                    g = t.readUnsignedExpGolomb(),
                    f = t.readUnsignedExpGolomb(),
                    _ = t.readUnsignedExpGolomb()),
                    t.readBoolean() && t.readBoolean()) {
                        switch (t.readUnsignedByte()) {
                        case 1:
                            y = [1, 1];
                            break;
                        case 2:
                            y = [12, 11];
                            break;
                        case 3:
                            y = [10, 11];
                            break;
                        case 4:
                            y = [16, 11];
                            break;
                        case 5:
                            y = [40, 33];
                            break;
                        case 6:
                            y = [24, 11];
                            break;
                        case 7:
                            y = [20, 11];
                            break;
                        case 8:
                            y = [32, 11];
                            break;
                        case 9:
                            y = [80, 33];
                            break;
                        case 10:
                            y = [18, 11];
                            break;
                        case 11:
                            y = [15, 11];
                            break;
                        case 12:
                            y = [64, 33];
                            break;
                        case 13:
                            y = [160, 99];
                            break;
                        case 14:
                            y = [4, 3];
                            break;
                        case 15:
                            y = [3, 2];
                            break;
                        case 16:
                            y = [2, 1];
                            break;
                        case 255:
                            y = [t.readUnsignedByte() << 8 | t.readUnsignedByte(), t.readUnsignedByte() << 8 | t.readUnsignedByte()]
                        }
                        y && (y[0],
                        y[1])
                    }
                    return {
                        profileIdc: i,
                        levelIdc: s,
                        profileCompatibility: n,
                        width: 16 * (h + 1) - 2 * m - 2 * g,
                        height: (2 - u) * (d + 1) * 16 - 2 * f - 2 * _,
                        sarRatio: y
                    }
                }
            }
            ).prototype = new ot;
            var ht, dt = {
                H264Stream: st,
                NalByteStream: nt
            }, ut = [96e3, 88200, 64e3, 48e3, 44100, 32e3, 24e3, 22050, 16e3, 12e3, 11025, 8e3, 7350], ct = function(e, t) {
                var i = e[t + 6] << 21 | e[t + 7] << 14 | e[t + 8] << 7 | e[t + 9];
                return i = i >= 0 ? i : 0,
                (16 & e[t + 5]) >> 4 ? i + 20 : i + 10
            }, pt = function(e, t) {
                return e.length - t < 10 || e[t] !== "I".charCodeAt(0) || e[t + 1] !== "D".charCodeAt(0) || e[t + 2] !== "3".charCodeAt(0) ? t : (t += ct(e, t),
                pt(e, t))
            }, mt = function(e) {
                return e[0] << 21 | e[1] << 14 | e[2] << 7 | e[3]
            }, gt = {
                isLikelyAacData: function(e) {
                    var t = pt(e, 0);
                    return e.length >= t + 2 && 255 === (255 & e[t]) && 240 === (240 & e[t + 1]) && 16 === (22 & e[t + 1])
                },
                parseId3TagSize: ct,
                parseAdtsSize: function(e, t) {
                    var i = (224 & e[t + 5]) >> 5
                      , s = e[t + 4] << 3;
                    return 6144 & e[t + 3] | s | i
                },
                parseType: function(e, t) {
                    return e[t] === "I".charCodeAt(0) && e[t + 1] === "D".charCodeAt(0) && e[t + 2] === "3".charCodeAt(0) ? "timed-metadata" : !0 & e[t] && 240 === (240 & e[t + 1]) ? "audio" : null
                },
                parseSampleRate: function(e) {
                    for (var t = 0; t + 5 < e.length; ) {
                        if (255 === e[t] && 240 === (246 & e[t + 1]))
                            return ut[(60 & e[t + 2]) >>> 2];
                        t++
                    }
                    return null
                },
                parseAacTimestamp: function(e) {
                    var t, i, s;
                    t = 10,
                    64 & e[5] && (t += 4,
                    t += mt(e.subarray(10, 14)));
                    do {
                        if ((i = mt(e.subarray(t + 4, t + 8))) < 1)
                            return null;
                        if ("PRIV" === String.fromCharCode(e[t], e[t + 1], e[t + 2], e[t + 3])) {
                            s = e.subarray(t + 10, t + i + 10);
                            for (var n = 0; n < s.byteLength; n++)
                                if (0 === s[n]) {
                                    if ("com.apple.streaming.transportStreamTimestamp" === unescape(function(e, t, i) {
                                        var s, n = "";
                                        for (s = t; s < i; s++)
                                            n += "%" + ("00" + e[s].toString(16)).slice(-2);
                                        return n
                                    }(s, 0, n))) {
                                        var r = s.subarray(n + 1)
                                          , a = (1 & r[3]) << 30 | r[4] << 22 | r[5] << 14 | r[6] << 6 | r[7] >>> 2;
                                        return a *= 4,
                                        a += 3 & r[7]
                                    }
                                    break
                                }
                        }
                        t += 10,
                        t += i
                    } while (t < e.byteLength);
                    return null
                }
            }, ft = gt;
            (ht = function() {
                var e = new Uint8Array
                  , t = 0;
                ht.prototype.init.call(this),
                this.setTimestamp = function(e) {
                    t = e
                }
                ,
                this.push = function(i) {
                    var s, n, r, a, o = 0, l = 0;
                    for (e.length ? (a = e.length,
                    (e = new Uint8Array(i.byteLength + a)).set(e.subarray(0, a)),
                    e.set(i, a)) : e = i; e.length - l >= 3; )
                        if (e[l] !== "I".charCodeAt(0) || e[l + 1] !== "D".charCodeAt(0) || e[l + 2] !== "3".charCodeAt(0))
                            if (255 !== (255 & e[l]) || 240 !== (240 & e[l + 1]))
                                l++;
                            else {
                                if (e.length - l < 7)
                                    break;
                                if (l + (o = ft.parseAdtsSize(e, l)) > e.length)
                                    break;
                                r = {
                                    type: "audio",
                                    data: e.subarray(l, l + o),
                                    pts: t,
                                    dts: t
                                },
                                this.trigger("data", r),
                                l += o
                            }
                        else {
                            if (e.length - l < 10)
                                break;
                            if (l + (o = ft.parseId3TagSize(e, l)) > e.length)
                                break;
                            n = {
                                type: "timed-metadata",
                                data: e.subarray(l, l + o)
                            },
                            this.trigger("data", n),
                            l += o
                        }
                    s = e.length - l,
                    e = s > 0 ? e.subarray(l) : new Uint8Array
                }
                ,
                this.reset = function() {
                    e = new Uint8Array,
                    this.trigger("reset")
                }
                ,
                this.endTimeline = function() {
                    e = new Uint8Array,
                    this.trigger("endedtimeline")
                }
            }
            ).prototype = new F;
            var _t, yt, vt, Tt, bt = F, St = Y, kt = Z, Ct = oe, Et = he, wt = et, xt = ne, It = at, Pt = dt.H264Stream, At = ht, Lt = gt.isLikelyAacData, Dt = ne.ONE_SECOND_IN_TS, Ot = ["audioobjecttype", "channelcount", "samplerate", "samplingfrequencyindex", "samplesize"], Mt = ["width", "height", "profileIdc", "levelIdc", "profileCompatibility", "sarRatio"], Rt = function(e, t) {
                t.stream = e,
                this.trigger("log", t)
            }, Ut = function(e, t) {
                for (var i = Object.keys(t), s = 0; s < i.length; s++) {
                    var n = i[s];
                    "headOfPipeline" !== n && t[n].on && t[n].on("log", Rt.bind(e, n))
                }
            }, Bt = function(e, t) {
                var i;
                if (e.length !== t.length)
                    return !1;
                for (i = 0; i < e.length; i++)
                    if (e[i] !== t[i])
                        return !1;
                return !0
            }, Nt = function(e, t, i, s, n, r) {
                return {
                    start: {
                        dts: e,
                        pts: e + (i - t)
                    },
                    end: {
                        dts: e + (s - t),
                        pts: e + (n - i)
                    },
                    prependedContentDuration: r,
                    baseMediaDecodeTime: e
                }
            };
            (yt = function(e, t) {
                var i, s = [], n = 0, r = 0, a = 1 / 0;
                i = (t = t || {}).firstSequenceNumber || 0,
                yt.prototype.init.call(this),
                this.push = function(t) {
                    Et.collectDtsInfo(e, t),
                    e && Ot.forEach((function(i) {
                        e[i] = t[i]
                    }
                    )),
                    s.push(t)
                }
                ,
                this.setEarliestDts = function(e) {
                    n = e
                }
                ,
                this.setVideoBaseMediaDecodeTime = function(e) {
                    a = e
                }
                ,
                this.setAudioAppendStart = function(e) {
                    r = e
                }
                ,
                this.flush = function() {
                    var o, l, h, d, u, c, p;
                    0 !== s.length ? (o = Ct.trimAdtsFramesByEarliestDts(s, e, n),
                    e.baseMediaDecodeTime = Et.calculateTrackBaseMediaDecodeTime(e, t.keepOriginalTimestamps),
                    p = Ct.prefixWithSilence(e, o, r, a),
                    e.samples = Ct.generateSampleTable(o),
                    h = St.mdat(Ct.concatenateFrameData(o)),
                    s = [],
                    l = St.moof(i, [e]),
                    d = new Uint8Array(l.byteLength + h.byteLength),
                    i++,
                    d.set(l),
                    d.set(h, l.byteLength),
                    Et.clearDtsInfo(e),
                    u = Math.ceil(1024 * Dt / e.samplerate),
                    o.length && (c = o.length * u,
                    this.trigger("segmentTimingInfo", Nt(xt.audioTsToVideoTs(e.baseMediaDecodeTime, e.samplerate), o[0].dts, o[0].pts, o[0].dts + c, o[0].pts + c, p || 0)),
                    this.trigger("timingInfo", {
                        start: o[0].pts,
                        end: o[0].pts + c
                    })),
                    this.trigger("data", {
                        track: e,
                        boxes: d
                    }),
                    this.trigger("done", "AudioSegmentStream")) : this.trigger("done", "AudioSegmentStream")
                }
                ,
                this.reset = function() {
                    Et.clearDtsInfo(e),
                    s = [],
                    this.trigger("reset")
                }
            }
            ).prototype = new bt,
            (_t = function(e, t) {
                var i, s, n, r = [], a = [];
                i = (t = t || {}).firstSequenceNumber || 0,
                _t.prototype.init.call(this),
                delete e.minPTS,
                this.gopCache_ = [],
                this.push = function(t) {
                    Et.collectDtsInfo(e, t),
                    "seq_parameter_set_rbsp" !== t.nalUnitType || s || (s = t.config,
                    e.sps = [t.data],
                    Mt.forEach((function(t) {
                        e[t] = s[t]
                    }
                    ), this)),
                    "pic_parameter_set_rbsp" !== t.nalUnitType || n || (n = t.data,
                    e.pps = [t.data]),
                    r.push(t)
                }
                ,
                this.flush = function() {
                    for (var s, n, o, l, h, d, u, c, p = 0; r.length && "access_unit_delimiter_rbsp" !== r[0].nalUnitType; )
                        r.shift();
                    if (0 === r.length)
                        return this.resetStream_(),
                        void this.trigger("done", "VideoSegmentStream");
                    if (s = kt.groupNalsIntoFrames(r),
                    (o = kt.groupFramesIntoGops(s))[0][0].keyFrame || ((n = this.getGopForFusion_(r[0], e)) ? (p = n.duration,
                    o.unshift(n),
                    o.byteLength += n.byteLength,
                    o.nalCount += n.nalCount,
                    o.pts = n.pts,
                    o.dts = n.dts,
                    o.duration += n.duration) : o = kt.extendFirstKeyFrame(o)),
                    a.length) {
                        var m;
                        if (!(m = t.alignGopsAtEnd ? this.alignGopsAtEnd_(o) : this.alignGopsAtStart_(o)))
                            return this.gopCache_.unshift({
                                gop: o.pop(),
                                pps: e.pps,
                                sps: e.sps
                            }),
                            this.gopCache_.length = Math.min(6, this.gopCache_.length),
                            r = [],
                            this.resetStream_(),
                            void this.trigger("done", "VideoSegmentStream");
                        Et.clearDtsInfo(e),
                        o = m
                    }
                    Et.collectDtsInfo(e, o),
                    e.samples = kt.generateSampleTable(o),
                    h = St.mdat(kt.concatenateNalData(o)),
                    e.baseMediaDecodeTime = Et.calculateTrackBaseMediaDecodeTime(e, t.keepOriginalTimestamps),
                    this.trigger("processedGopsInfo", o.map((function(e) {
                        return {
                            pts: e.pts,
                            dts: e.dts,
                            byteLength: e.byteLength
                        }
                    }
                    ))),
                    u = o[0],
                    c = o[o.length - 1],
                    this.trigger("segmentTimingInfo", Nt(e.baseMediaDecodeTime, u.dts, u.pts, c.dts + c.duration, c.pts + c.duration, p)),
                    this.trigger("timingInfo", {
                        start: o[0].pts,
                        end: o[o.length - 1].pts + o[o.length - 1].duration
                    }),
                    this.gopCache_.unshift({
                        gop: o.pop(),
                        pps: e.pps,
                        sps: e.sps
                    }),
                    this.gopCache_.length = Math.min(6, this.gopCache_.length),
                    r = [],
                    this.trigger("baseMediaDecodeTime", e.baseMediaDecodeTime),
                    this.trigger("timelineStartInfo", e.timelineStartInfo),
                    l = St.moof(i, [e]),
                    d = new Uint8Array(l.byteLength + h.byteLength),
                    i++,
                    d.set(l),
                    d.set(h, l.byteLength),
                    this.trigger("data", {
                        track: e,
                        boxes: d
                    }),
                    this.resetStream_(),
                    this.trigger("done", "VideoSegmentStream")
                }
                ,
                this.reset = function() {
                    this.resetStream_(),
                    r = [],
                    this.gopCache_.length = 0,
                    a.length = 0,
                    this.trigger("reset")
                }
                ,
                this.resetStream_ = function() {
                    Et.clearDtsInfo(e),
                    s = void 0,
                    n = void 0
                }
                ,
                this.getGopForFusion_ = function(t) {
                    var i, s, n, r, a, o = 1 / 0;
                    for (a = 0; a < this.gopCache_.length; a++)
                        n = (r = this.gopCache_[a]).gop,
                        e.pps && Bt(e.pps[0], r.pps[0]) && e.sps && Bt(e.sps[0], r.sps[0]) && (n.dts < e.timelineStartInfo.dts || (i = t.dts - n.dts - n.duration) >= -1e4 && i <= 45e3 && (!s || o > i) && (s = r,
                        o = i));
                    return s ? s.gop : null
                }
                ,
                this.alignGopsAtStart_ = function(e) {
                    var t, i, s, n, r, o, l, h;
                    for (r = e.byteLength,
                    o = e.nalCount,
                    l = e.duration,
                    t = i = 0; t < a.length && i < e.length && (s = a[t],
                    n = e[i],
                    s.pts !== n.pts); )
                        n.pts > s.pts ? t++ : (i++,
                        r -= n.byteLength,
                        o -= n.nalCount,
                        l -= n.duration);
                    return 0 === i ? e : i === e.length ? null : ((h = e.slice(i)).byteLength = r,
                    h.duration = l,
                    h.nalCount = o,
                    h.pts = h[0].pts,
                    h.dts = h[0].dts,
                    h)
                }
                ,
                this.alignGopsAtEnd_ = function(e) {
                    var t, i, s, n, r, o, l;
                    for (t = a.length - 1,
                    i = e.length - 1,
                    r = null,
                    o = !1; t >= 0 && i >= 0; ) {
                        if (s = a[t],
                        n = e[i],
                        s.pts === n.pts) {
                            o = !0;
                            break
                        }
                        s.pts > n.pts ? t-- : (t === a.length - 1 && (r = i),
                        i--)
                    }
                    if (!o && null === r)
                        return null;
                    if (0 === (l = o ? i : r))
                        return e;
                    var h = e.slice(l)
                      , d = h.reduce((function(e, t) {
                        return e.byteLength += t.byteLength,
                        e.duration += t.duration,
                        e.nalCount += t.nalCount,
                        e
                    }
                    ), {
                        byteLength: 0,
                        duration: 0,
                        nalCount: 0
                    });
                    return h.byteLength = d.byteLength,
                    h.duration = d.duration,
                    h.nalCount = d.nalCount,
                    h.pts = h[0].pts,
                    h.dts = h[0].dts,
                    h
                }
                ,
                this.alignGopsWith = function(e) {
                    a = e
                }
            }
            ).prototype = new bt,
            (Tt = function(e, t) {
                this.numberOfTracks = 0,
                this.metadataStream = t,
                "undefined" !== typeof (e = e || {}).remux ? this.remuxTracks = !!e.remux : this.remuxTracks = !0,
                "boolean" === typeof e.keepOriginalTimestamps ? this.keepOriginalTimestamps = e.keepOriginalTimestamps : this.keepOriginalTimestamps = !1,
                this.pendingTracks = [],
                this.videoTrack = null,
                this.pendingBoxes = [],
                this.pendingCaptions = [],
                this.pendingMetadata = [],
                this.pendingBytes = 0,
                this.emittedTracks = 0,
                Tt.prototype.init.call(this),
                this.push = function(e) {
                    return e.content || e.text ? this.pendingCaptions.push(e) : e.frames ? this.pendingMetadata.push(e) : (this.pendingTracks.push(e.track),
                    this.pendingBytes += e.boxes.byteLength,
                    "video" === e.track.type && (this.videoTrack = e.track,
                    this.pendingBoxes.push(e.boxes)),
                    void ("audio" === e.track.type && (this.audioTrack = e.track,
                    this.pendingBoxes.unshift(e.boxes))))
                }
            }
            ).prototype = new bt,
            Tt.prototype.flush = function(e) {
                var t, i, s, n, r = 0, a = {
                    captions: [],
                    captionStreams: {},
                    metadata: [],
                    info: {}
                }, o = 0;
                if (this.pendingTracks.length < this.numberOfTracks) {
                    if ("VideoSegmentStream" !== e && "AudioSegmentStream" !== e)
                        return;
                    if (this.remuxTracks)
                        return;
                    if (0 === this.pendingTracks.length)
                        return this.emittedTracks++,
                        void (this.emittedTracks >= this.numberOfTracks && (this.trigger("done"),
                        this.emittedTracks = 0))
                }
                if (this.videoTrack ? (o = this.videoTrack.timelineStartInfo.pts,
                Mt.forEach((function(e) {
                    a.info[e] = this.videoTrack[e]
                }
                ), this)) : this.audioTrack && (o = this.audioTrack.timelineStartInfo.pts,
                Ot.forEach((function(e) {
                    a.info[e] = this.audioTrack[e]
                }
                ), this)),
                this.videoTrack || this.audioTrack) {
                    for (1 === this.pendingTracks.length ? a.type = this.pendingTracks[0].type : a.type = "combined",
                    this.emittedTracks += this.pendingTracks.length,
                    s = St.initSegment(this.pendingTracks),
                    a.initSegment = new Uint8Array(s.byteLength),
                    a.initSegment.set(s),
                    a.data = new Uint8Array(this.pendingBytes),
                    n = 0; n < this.pendingBoxes.length; n++)
                        a.data.set(this.pendingBoxes[n], r),
                        r += this.pendingBoxes[n].byteLength;
                    for (n = 0; n < this.pendingCaptions.length; n++)
                        (t = this.pendingCaptions[n]).startTime = xt.metadataTsToSeconds(t.startPts, o, this.keepOriginalTimestamps),
                        t.endTime = xt.metadataTsToSeconds(t.endPts, o, this.keepOriginalTimestamps),
                        a.captionStreams[t.stream] = !0,
                        a.captions.push(t);
                    for (n = 0; n < this.pendingMetadata.length; n++)
                        (i = this.pendingMetadata[n]).cueTime = xt.metadataTsToSeconds(i.pts, o, this.keepOriginalTimestamps),
                        a.metadata.push(i);
                    for (a.metadata.dispatchType = this.metadataStream.dispatchType,
                    this.pendingTracks.length = 0,
                    this.videoTrack = null,
                    this.pendingBoxes.length = 0,
                    this.pendingCaptions.length = 0,
                    this.pendingBytes = 0,
                    this.pendingMetadata.length = 0,
                    this.trigger("data", a),
                    n = 0; n < a.captions.length; n++)
                        t = a.captions[n],
                        this.trigger("caption", t);
                    for (n = 0; n < a.metadata.length; n++)
                        i = a.metadata[n],
                        this.trigger("id3Frame", i)
                }
                this.emittedTracks >= this.numberOfTracks && (this.trigger("done"),
                this.emittedTracks = 0)
            }
            ,
            Tt.prototype.setRemux = function(e) {
                this.remuxTracks = e
            }
            ,
            (vt = function(e) {
                var t, i, s = this, n = !0;
                vt.prototype.init.call(this),
                e = e || {},
                this.baseMediaDecodeTime = e.baseMediaDecodeTime || 0,
                this.transmuxPipeline_ = {},
                this.setupAacPipeline = function() {
                    var n = {};
                    this.transmuxPipeline_ = n,
                    n.type = "aac",
                    n.metadataStream = new wt.MetadataStream,
                    n.aacStream = new At,
                    n.audioTimestampRolloverStream = new wt.TimestampRolloverStream("audio"),
                    n.timedMetadataTimestampRolloverStream = new wt.TimestampRolloverStream("timed-metadata"),
                    n.adtsStream = new It,
                    n.coalesceStream = new Tt(e,n.metadataStream),
                    n.headOfPipeline = n.aacStream,
                    n.aacStream.pipe(n.audioTimestampRolloverStream).pipe(n.adtsStream),
                    n.aacStream.pipe(n.timedMetadataTimestampRolloverStream).pipe(n.metadataStream).pipe(n.coalesceStream),
                    n.metadataStream.on("timestamp", (function(e) {
                        n.aacStream.setTimestamp(e.timeStamp)
                    }
                    )),
                    n.aacStream.on("data", (function(r) {
                        "timed-metadata" !== r.type && "audio" !== r.type || n.audioSegmentStream || (i = i || {
                            timelineStartInfo: {
                                baseMediaDecodeTime: s.baseMediaDecodeTime
                            },
                            codec: "adts",
                            type: "audio"
                        },
                        n.coalesceStream.numberOfTracks++,
                        n.audioSegmentStream = new yt(i,e),
                        n.audioSegmentStream.on("log", s.getLogTrigger_("audioSegmentStream")),
                        n.audioSegmentStream.on("timingInfo", s.trigger.bind(s, "audioTimingInfo")),
                        n.adtsStream.pipe(n.audioSegmentStream).pipe(n.coalesceStream),
                        s.trigger("trackinfo", {
                            hasAudio: !!i,
                            hasVideo: !!t
                        }))
                    }
                    )),
                    n.coalesceStream.on("data", this.trigger.bind(this, "data")),
                    n.coalesceStream.on("done", this.trigger.bind(this, "done")),
                    Ut(this, n)
                }
                ,
                this.setupTsPipeline = function() {
                    var n = {};
                    this.transmuxPipeline_ = n,
                    n.type = "ts",
                    n.metadataStream = new wt.MetadataStream,
                    n.packetStream = new wt.TransportPacketStream,
                    n.parseStream = new wt.TransportParseStream,
                    n.elementaryStream = new wt.ElementaryStream,
                    n.timestampRolloverStream = new wt.TimestampRolloverStream,
                    n.adtsStream = new It,
                    n.h264Stream = new Pt,
                    n.captionStream = new wt.CaptionStream(e),
                    n.coalesceStream = new Tt(e,n.metadataStream),
                    n.headOfPipeline = n.packetStream,
                    n.packetStream.pipe(n.parseStream).pipe(n.elementaryStream).pipe(n.timestampRolloverStream),
                    n.timestampRolloverStream.pipe(n.h264Stream),
                    n.timestampRolloverStream.pipe(n.adtsStream),
                    n.timestampRolloverStream.pipe(n.metadataStream).pipe(n.coalesceStream),
                    n.h264Stream.pipe(n.captionStream).pipe(n.coalesceStream),
                    n.elementaryStream.on("data", (function(r) {
                        var a;
                        if ("metadata" === r.type) {
                            for (a = r.tracks.length; a--; )
                                t || "video" !== r.tracks[a].type ? i || "audio" !== r.tracks[a].type || ((i = r.tracks[a]).timelineStartInfo.baseMediaDecodeTime = s.baseMediaDecodeTime) : (t = r.tracks[a]).timelineStartInfo.baseMediaDecodeTime = s.baseMediaDecodeTime;
                            t && !n.videoSegmentStream && (n.coalesceStream.numberOfTracks++,
                            n.videoSegmentStream = new _t(t,e),
                            n.videoSegmentStream.on("log", s.getLogTrigger_("videoSegmentStream")),
                            n.videoSegmentStream.on("timelineStartInfo", (function(t) {
                                i && !e.keepOriginalTimestamps && (i.timelineStartInfo = t,
                                n.audioSegmentStream.setEarliestDts(t.dts - s.baseMediaDecodeTime))
                            }
                            )),
                            n.videoSegmentStream.on("processedGopsInfo", s.trigger.bind(s, "gopInfo")),
                            n.videoSegmentStream.on("segmentTimingInfo", s.trigger.bind(s, "videoSegmentTimingInfo")),
                            n.videoSegmentStream.on("baseMediaDecodeTime", (function(e) {
                                i && n.audioSegmentStream.setVideoBaseMediaDecodeTime(e)
                            }
                            )),
                            n.videoSegmentStream.on("timingInfo", s.trigger.bind(s, "videoTimingInfo")),
                            n.h264Stream.pipe(n.videoSegmentStream).pipe(n.coalesceStream)),
                            i && !n.audioSegmentStream && (n.coalesceStream.numberOfTracks++,
                            n.audioSegmentStream = new yt(i,e),
                            n.audioSegmentStream.on("log", s.getLogTrigger_("audioSegmentStream")),
                            n.audioSegmentStream.on("timingInfo", s.trigger.bind(s, "audioTimingInfo")),
                            n.audioSegmentStream.on("segmentTimingInfo", s.trigger.bind(s, "audioSegmentTimingInfo")),
                            n.adtsStream.pipe(n.audioSegmentStream).pipe(n.coalesceStream)),
                            s.trigger("trackinfo", {
                                hasAudio: !!i,
                                hasVideo: !!t
                            })
                        }
                    }
                    )),
                    n.coalesceStream.on("data", this.trigger.bind(this, "data")),
                    n.coalesceStream.on("id3Frame", (function(e) {
                        e.dispatchType = n.metadataStream.dispatchType,
                        s.trigger("id3Frame", e)
                    }
                    )),
                    n.coalesceStream.on("caption", this.trigger.bind(this, "caption")),
                    n.coalesceStream.on("done", this.trigger.bind(this, "done")),
                    Ut(this, n)
                }
                ,
                this.setBaseMediaDecodeTime = function(s) {
                    var n = this.transmuxPipeline_;
                    e.keepOriginalTimestamps || (this.baseMediaDecodeTime = s),
                    i && (i.timelineStartInfo.dts = void 0,
                    i.timelineStartInfo.pts = void 0,
                    Et.clearDtsInfo(i),
                    n.audioTimestampRolloverStream && n.audioTimestampRolloverStream.discontinuity()),
                    t && (n.videoSegmentStream && (n.videoSegmentStream.gopCache_ = []),
                    t.timelineStartInfo.dts = void 0,
                    t.timelineStartInfo.pts = void 0,
                    Et.clearDtsInfo(t),
                    n.captionStream.reset()),
                    n.timestampRolloverStream && n.timestampRolloverStream.discontinuity()
                }
                ,
                this.setAudioAppendStart = function(e) {
                    i && this.transmuxPipeline_.audioSegmentStream.setAudioAppendStart(e)
                }
                ,
                this.setRemux = function(t) {
                    var i = this.transmuxPipeline_;
                    e.remux = t,
                    i && i.coalesceStream && i.coalesceStream.setRemux(t)
                }
                ,
                this.alignGopsWith = function(e) {
                    t && this.transmuxPipeline_.videoSegmentStream && this.transmuxPipeline_.videoSegmentStream.alignGopsWith(e)
                }
                ,
                this.getLogTrigger_ = function(e) {
                    var t = this;
                    return function(i) {
                        i.stream = e,
                        t.trigger("log", i)
                    }
                }
                ,
                this.push = function(e) {
                    if (n) {
                        var t = Lt(e);
                        t && "aac" !== this.transmuxPipeline_.type ? this.setupAacPipeline() : t || "ts" === this.transmuxPipeline_.type || this.setupTsPipeline(),
                        n = !1
                    }
                    this.transmuxPipeline_.headOfPipeline.push(e)
                }
                ,
                this.flush = function() {
                    n = !0,
                    this.transmuxPipeline_.headOfPipeline.flush()
                }
                ,
                this.endTimeline = function() {
                    this.transmuxPipeline_.headOfPipeline.endTimeline()
                }
                ,
                this.reset = function() {
                    this.transmuxPipeline_.headOfPipeline && this.transmuxPipeline_.headOfPipeline.reset()
                }
                ,
                this.resetCaptions = function() {
                    this.transmuxPipeline_.captionStream && this.transmuxPipeline_.captionStream.reset()
                }
            }
            ).prototype = new bt;
            var Ft, jt, $t, qt, Ht, Vt = {
                Transmuxer: vt,
                VideoSegmentStream: _t,
                AudioSegmentStream: yt,
                AUDIO_PROPERTIES: Ot,
                VIDEO_PROPERTIES: Mt,
                generateSegmentTimingInfo: Nt
            }, zt = function(e) {
                return e >>> 0
            }, Wt = function(e) {
                var t = "";
                return t += String.fromCharCode(e[0]),
                t += String.fromCharCode(e[1]),
                t += String.fromCharCode(e[2]),
                t += String.fromCharCode(e[3])
            }, Gt = zt, Kt = Wt, Qt = function(e, t) {
                var i, s, n, r, a, o = [];
                if (!t.length)
                    return null;
                for (i = 0; i < e.byteLength; )
                    s = Gt(e[i] << 24 | e[i + 1] << 16 | e[i + 2] << 8 | e[i + 3]),
                    n = Kt(e.subarray(i + 4, i + 8)),
                    r = s > 1 ? i + s : e.byteLength,
                    n === t[0] && (1 === t.length ? o.push(e.subarray(i + 8, r)) : (a = Qt(e.subarray(i + 8, r), t.slice(1))).length && (o = o.concat(a))),
                    i = r;
                return o
            }, Xt = zt, Yt = $.getUint64, Jt = function(e) {
                var t = {
                    version: e[0],
                    flags: new Uint8Array(e.subarray(1, 4))
                };
                return 1 === t.version ? t.baseMediaDecodeTime = Yt(e.subarray(4)) : t.baseMediaDecodeTime = Xt(e[4] << 24 | e[5] << 16 | e[6] << 8 | e[7]),
                t
            }, Zt = function(e) {
                return {
                    isLeading: (12 & e[0]) >>> 2,
                    dependsOn: 3 & e[0],
                    isDependedOn: (192 & e[1]) >>> 6,
                    hasRedundancy: (48 & e[1]) >>> 4,
                    paddingValue: (14 & e[1]) >>> 1,
                    isNonSyncSample: 1 & e[1],
                    degradationPriority: e[2] << 8 | e[3]
                }
            }, ei = function(e) {
                var t, i = {
                    version: e[0],
                    flags: new Uint8Array(e.subarray(1, 4)),
                    samples: []
                }, s = new DataView(e.buffer,e.byteOffset,e.byteLength), n = 1 & i.flags[2], r = 4 & i.flags[2], a = 1 & i.flags[1], o = 2 & i.flags[1], l = 4 & i.flags[1], h = 8 & i.flags[1], d = s.getUint32(4), u = 8;
                for (n && (i.dataOffset = s.getInt32(u),
                u += 4),
                r && d && (t = {
                    flags: Zt(e.subarray(u, u + 4))
                },
                u += 4,
                a && (t.duration = s.getUint32(u),
                u += 4),
                o && (t.size = s.getUint32(u),
                u += 4),
                h && (1 === i.version ? t.compositionTimeOffset = s.getInt32(u) : t.compositionTimeOffset = s.getUint32(u),
                u += 4),
                i.samples.push(t),
                d--); d--; )
                    t = {},
                    a && (t.duration = s.getUint32(u),
                    u += 4),
                    o && (t.size = s.getUint32(u),
                    u += 4),
                    l && (t.flags = Zt(e.subarray(u, u + 4)),
                    u += 4),
                    h && (1 === i.version ? t.compositionTimeOffset = s.getInt32(u) : t.compositionTimeOffset = s.getUint32(u),
                    u += 4),
                    i.samples.push(t);
                return i
            }, ti = function(e) {
                var t, i = new DataView(e.buffer,e.byteOffset,e.byteLength), s = {
                    version: e[0],
                    flags: new Uint8Array(e.subarray(1, 4)),
                    trackId: i.getUint32(4)
                }, n = 1 & s.flags[2], r = 2 & s.flags[2], a = 8 & s.flags[2], o = 16 & s.flags[2], l = 32 & s.flags[2], h = 65536 & s.flags[0], d = 131072 & s.flags[0];
                return t = 8,
                n && (t += 4,
                s.baseDataOffset = i.getUint32(12),
                t += 4),
                r && (s.sampleDescriptionIndex = i.getUint32(t),
                t += 4),
                a && (s.defaultSampleDuration = i.getUint32(t),
                t += 4),
                o && (s.defaultSampleSize = i.getUint32(t),
                t += 4),
                l && (s.defaultSampleFlags = i.getUint32(t)),
                h && (s.durationIsEmpty = !0),
                !n && d && (s.baseDataOffsetIsMoof = !0),
                s
            }, ii = (Ft = "undefined" !== typeof window ? window : "undefined" !== typeof e ? e : "undefined" !== typeof self ? self : {},
            de.discardEmulationPreventionBytes), si = Ce.CaptionStream, ni = Qt, ri = Jt, ai = ei, oi = ti, li = Ft, hi = function(e, t) {
                for (var i = e, s = 0; s < t.length; s++) {
                    var n = t[s];
                    if (i < n.size)
                        return n;
                    i -= n.size
                }
                return null
            }, di = function(e, t) {
                var i = ni(e, ["moof", "traf"])
                  , s = ni(e, ["mdat"])
                  , n = {}
                  , r = [];
                return s.forEach((function(e, t) {
                    var s = i[t];
                    r.push({
                        mdat: e,
                        traf: s
                    })
                }
                )),
                r.forEach((function(e) {
                    var i, s = e.mdat, r = e.traf, a = ni(r, ["tfhd"]), o = oi(a[0]), l = o.trackId, h = ni(r, ["tfdt"]), d = h.length > 0 ? ri(h[0]).baseMediaDecodeTime : 0, u = ni(r, ["trun"]);
                    t === l && u.length > 0 && (i = function(e, t, i) {
                        var s, n, r, a, o = new DataView(e.buffer,e.byteOffset,e.byteLength), l = {
                            logs: [],
                            seiNals: []
                        };
                        for (n = 0; n + 4 < e.length; n += r)
                            if (r = o.getUint32(n),
                            n += 4,
                            !(r <= 0))
                                switch (31 & e[n]) {
                                case 6:
                                    var h = e.subarray(n + 1, n + 1 + r)
                                      , d = hi(n, t);
                                    if (s = {
                                        nalUnitType: "sei_rbsp",
                                        size: r,
                                        data: h,
                                        escapedRBSP: ii(h),
                                        trackId: i
                                    },
                                    d)
                                        s.pts = d.pts,
                                        s.dts = d.dts,
                                        a = d;
                                    else {
                                        if (!a) {
                                            l.logs.push({
                                                level: "warn",
                                                message: "We've encountered a nal unit without data at " + n + " for trackId " + i + ". See mux.js#223."
                                            });
                                            break
                                        }
                                        s.pts = a.pts,
                                        s.dts = a.dts
                                    }
                                    l.seiNals.push(s)
                                }
                        return l
                    }(s, function(e, t, i) {
                        var s = t
                          , n = i.defaultSampleDuration || 0
                          , r = i.defaultSampleSize || 0
                          , a = i.trackId
                          , o = [];
                        return e.forEach((function(e) {
                            var t = ai(e).samples;
                            t.forEach((function(e) {
                                void 0 === e.duration && (e.duration = n),
                                void 0 === e.size && (e.size = r),
                                e.trackId = a,
                                e.dts = s,
                                void 0 === e.compositionTimeOffset && (e.compositionTimeOffset = 0),
                                "bigint" === typeof s ? (e.pts = s + li.BigInt(e.compositionTimeOffset),
                                s += li.BigInt(e.duration)) : (e.pts = s + e.compositionTimeOffset,
                                s += e.duration)
                            }
                            )),
                            o = o.concat(t)
                        }
                        )),
                        o
                    }(u, d, o), l),
                    n[l] || (n[l] = {
                        seiNals: [],
                        logs: []
                    }),
                    n[l].seiNals = n[l].seiNals.concat(i.seiNals),
                    n[l].logs = n[l].logs.concat(i.logs))
                }
                )),
                n
            }, ui = function() {
                var e, t, i, s, n, r, a = !1;
                this.isInitialized = function() {
                    return a
                }
                ,
                this.init = function(t) {
                    e = new si,
                    a = !0,
                    r = !!t && t.isPartial,
                    e.on("data", (function(e) {
                        e.startTime = e.startPts / s,
                        e.endTime = e.endPts / s,
                        n.captions.push(e),
                        n.captionStreams[e.stream] = !0
                    }
                    )),
                    e.on("log", (function(e) {
                        n.logs.push(e)
                    }
                    ))
                }
                ,
                this.isNewInit = function(e, t) {
                    return !(e && 0 === e.length || t && "object" === typeof t && 0 === Object.keys(t).length) && (i !== e[0] || s !== t[i])
                }
                ,
                this.parse = function(e, r, a) {
                    var o;
                    if (!this.isInitialized())
                        return null;
                    if (!r || !a)
                        return null;
                    if (this.isNewInit(r, a))
                        i = r[0],
                        s = a[i];
                    else if (null === i || !s)
                        return t.push(e),
                        null;
                    for (; t.length > 0; ) {
                        var l = t.shift();
                        this.parse(l, r, a)
                    }
                    return (o = function(e, t, i) {
                        if (null === t)
                            return null;
                        var s = di(e, t)[t] || {};
                        return {
                            seiNals: s.seiNals,
                            logs: s.logs,
                            timescale: i
                        }
                    }(e, i, s)) && o.logs && (n.logs = n.logs.concat(o.logs)),
                    null !== o && o.seiNals ? (this.pushNals(o.seiNals),
                    this.flushStream(),
                    n) : n.logs.length ? {
                        logs: n.logs,
                        captions: [],
                        captionStreams: []
                    } : null
                }
                ,
                this.pushNals = function(t) {
                    if (!this.isInitialized() || !t || 0 === t.length)
                        return null;
                    t.forEach((function(t) {
                        e.push(t)
                    }
                    ))
                }
                ,
                this.flushStream = function() {
                    if (!this.isInitialized())
                        return null;
                    r ? e.partialFlush() : e.flush()
                }
                ,
                this.clearParsedCaptions = function() {
                    n.captions = [],
                    n.captionStreams = {},
                    n.logs = []
                }
                ,
                this.resetCaptionStream = function() {
                    if (!this.isInitialized())
                        return null;
                    e.reset()
                }
                ,
                this.clearAllCaptions = function() {
                    this.clearParsedCaptions(),
                    this.resetCaptionStream()
                }
                ,
                this.reset = function() {
                    t = [],
                    i = null,
                    s = null,
                    n ? this.clearParsedCaptions() : n = {
                        captions: [],
                        captionStreams: {},
                        logs: []
                    },
                    this.resetCaptionStream()
                }
                ,
                this.reset()
            }, ci = function(e) {
                for (var t = 0, i = String.fromCharCode(e[t]), s = ""; "\0" !== i; )
                    s += i,
                    t++,
                    i = String.fromCharCode(e[t]);
                return s += i
            }, pi = $.getUint64, mi = function(e, t) {
                var i = "\0" !== t.scheme_id_uri
                  , s = 0 === e && gi(t.presentation_time_delta) && i
                  , n = 1 === e && gi(t.presentation_time) && i;
                return !(e > 1) && s || n
            }, gi = function(e) {
                return void 0 !== e || null !== e
            }, fi = zt, _i = function(e) {
                return ("00" + e.toString(16)).slice(-2)
            }, yi = Qt, vi = Wt, Ti = {
                parseEmsgBox: function(e) {
                    var t, i, s, n, r, a, o, l = 4, h = e[0];
                    if (0 === h)
                        l += (t = ci(e.subarray(l))).length,
                        l += (i = ci(e.subarray(l))).length,
                        s = (d = new DataView(e.buffer)).getUint32(l),
                        l += 4,
                        r = d.getUint32(l),
                        l += 4,
                        a = d.getUint32(l),
                        l += 4,
                        o = d.getUint32(l),
                        l += 4;
                    else if (1 === h) {
                        var d;
                        s = (d = new DataView(e.buffer)).getUint32(l),
                        l += 4,
                        n = pi(e.subarray(l)),
                        l += 8,
                        a = d.getUint32(l),
                        l += 4,
                        o = d.getUint32(l),
                        l += 4,
                        l += (t = ci(e.subarray(l))).length,
                        l += (i = ci(e.subarray(l))).length
                    }
                    var u = {
                        scheme_id_uri: t,
                        value: i,
                        timescale: s || 1,
                        presentation_time: n,
                        presentation_time_delta: r,
                        event_duration: a,
                        id: o,
                        message_data: new Uint8Array(e.subarray(l, e.byteLength))
                    };
                    return mi(h, u) ? u : void 0
                },
                scaleTime: function(e, t, i, s) {
                    return e || 0 === e ? e / t : s + i / t
                }
            }, bi = ti, Si = ei, ki = Jt, Ci = $.getUint64, Ei = Ft, wi = Fe.parseId3Frames;
            jt = function(e, t) {
                var i = yi(t, ["moof", "traf"]).reduce((function(t, i) {
                    var s, n = yi(i, ["tfhd"])[0], r = fi(n[4] << 24 | n[5] << 16 | n[6] << 8 | n[7]), a = e[r] || 9e4, o = yi(i, ["tfdt"])[0], l = new DataView(o.buffer,o.byteOffset,o.byteLength);
                    let h;
                    return "bigint" === typeof (s = 1 === o[0] ? Ci(o.subarray(4, 12)) : l.getUint32(4)) ? h = s / Ei.BigInt(a) : "number" !== typeof s || isNaN(s) || (h = s / a),
                    h < Number.MAX_SAFE_INTEGER && (h = Number(h)),
                    h < t && (t = h),
                    t
                }
                ), 1 / 0);
                return "bigint" === typeof i || isFinite(i) ? i : 0
            }
            ,
            $t = function(e) {
                var t = yi(e, ["moov", "trak"])
                  , i = [];
                return t.forEach((function(e) {
                    var t, s, n = {}, r = yi(e, ["tkhd"])[0];
                    r && (s = (t = new DataView(r.buffer,r.byteOffset,r.byteLength)).getUint8(0),
                    n.id = 0 === s ? t.getUint32(12) : t.getUint32(20));
                    var a = yi(e, ["mdia", "hdlr"])[0];
                    if (a) {
                        var o = vi(a.subarray(8, 12));
                        n.type = "vide" === o ? "video" : "soun" === o ? "audio" : o
                    }
                    var l = yi(e, ["mdia", "minf", "stbl", "stsd"])[0];
                    if (l) {
                        var h = l.subarray(8);
                        n.codec = vi(h.subarray(4, 8));
                        var d, u = yi(h, [n.codec])[0];
                        u && (/^[asm]vc[1-9]$/i.test(n.codec) ? (d = u.subarray(78),
                        "avcC" === vi(d.subarray(4, 8)) && d.length > 11 ? (n.codec += ".",
                        n.codec += _i(d[9]),
                        n.codec += _i(d[10]),
                        n.codec += _i(d[11])) : n.codec = "avc1.4d400d") : /^mp4[a,v]$/i.test(n.codec) ? (d = u.subarray(28),
                        "esds" === vi(d.subarray(4, 8)) && d.length > 20 && 0 !== d[19] ? (n.codec += "." + _i(d[19]),
                        n.codec += "." + _i(d[20] >>> 2 & 63).replace(/^0/, "")) : n.codec = "mp4a.40.2") : n.codec = n.codec.toLowerCase())
                    }
                    var c = yi(e, ["mdia", "mdhd"])[0];
                    c && (n.timescale = qt(c)),
                    i.push(n)
                }
                )),
                i
            }
            ,
            Ht = function(e, t=0) {
                return yi(e, ["emsg"]).map((e=>{
                    var i = Ti.parseEmsgBox(new Uint8Array(e))
                      , s = wi(i.message_data);
                    return {
                        cueTime: Ti.scaleTime(i.presentation_time, i.timescale, i.presentation_time_delta, t),
                        duration: Ti.scaleTime(i.event_duration, i.timescale),
                        frames: s
                    }
                }
                ))
            }
            ;
            var xi = jt
              , Ii = $t
              , Pi = (qt = function(e) {
                var t = 0 === e[0] ? 12 : 20;
                return fi(e[t] << 24 | e[t + 1] << 16 | e[t + 2] << 8 | e[t + 3])
            }
            ,
            Ht)
              , Ai = Ee
              , Li = function(e) {
                var t = 31 & e[1];
                return t <<= 8,
                t |= e[2]
            }
              , Di = function(e) {
                return !!(64 & e[1])
            }
              , Oi = function(e) {
                var t = 0;
                return (48 & e[3]) >>> 4 > 1 && (t += e[4] + 1),
                t
            }
              , Mi = function(e) {
                switch (e) {
                case 5:
                    return "slice_layer_without_partitioning_rbsp_idr";
                case 6:
                    return "sei_rbsp";
                case 7:
                    return "seq_parameter_set_rbsp";
                case 8:
                    return "pic_parameter_set_rbsp";
                case 9:
                    return "access_unit_delimiter_rbsp";
                default:
                    return null
                }
            }
              , Ri = {
                parseType: function(e, t) {
                    var i = Li(e);
                    return 0 === i ? "pat" : i === t ? "pmt" : t ? "pes" : null
                },
                parsePat: function(e) {
                    var t = Di(e)
                      , i = 4 + Oi(e);
                    return t && (i += e[i] + 1),
                    (31 & e[i + 10]) << 8 | e[i + 11]
                },
                parsePmt: function(e) {
                    var t = {}
                      , i = Di(e)
                      , s = 4 + Oi(e);
                    if (i && (s += e[s] + 1),
                    1 & e[s + 5]) {
                        var n;
                        n = 3 + ((15 & e[s + 1]) << 8 | e[s + 2]) - 4;
                        for (var r = 12 + ((15 & e[s + 10]) << 8 | e[s + 11]); r < n; ) {
                            var a = s + r;
                            t[(31 & e[a + 1]) << 8 | e[a + 2]] = e[a],
                            r += 5 + ((15 & e[a + 3]) << 8 | e[a + 4])
                        }
                        return t
                    }
                },
                parsePayloadUnitStartIndicator: Di,
                parsePesType: function(e, t) {
                    switch (t[Li(e)]) {
                    case Ai.H264_STREAM_TYPE:
                        return "video";
                    case Ai.ADTS_STREAM_TYPE:
                        return "audio";
                    case Ai.METADATA_STREAM_TYPE:
                        return "timed-metadata";
                    default:
                        return null
                    }
                },
                parsePesTime: function(e) {
                    if (!Di(e))
                        return null;
                    var t = 4 + Oi(e);
                    if (t >= e.byteLength)
                        return null;
                    var i, s = null;
                    return 192 & (i = e[t + 7]) && ((s = {}).pts = (14 & e[t + 9]) << 27 | (255 & e[t + 10]) << 20 | (254 & e[t + 11]) << 12 | (255 & e[t + 12]) << 5 | (254 & e[t + 13]) >>> 3,
                    s.pts *= 4,
                    s.pts += (6 & e[t + 13]) >>> 1,
                    s.dts = s.pts,
                    64 & i && (s.dts = (14 & e[t + 14]) << 27 | (255 & e[t + 15]) << 20 | (254 & e[t + 16]) << 12 | (255 & e[t + 17]) << 5 | (254 & e[t + 18]) >>> 3,
                    s.dts *= 4,
                    s.dts += (6 & e[t + 18]) >>> 1)),
                    s
                },
                videoPacketContainsKeyFrame: function(e) {
                    for (var t = 4 + Oi(e), i = e.subarray(t), s = 0, n = 0, r = !1; n < i.byteLength - 3; n++)
                        if (1 === i[n + 2]) {
                            s = n + 5;
                            break
                        }
                    for (; s < i.byteLength; )
                        switch (i[s]) {
                        case 0:
                            if (0 !== i[s - 1]) {
                                s += 2;
                                break
                            }
                            if (0 !== i[s - 2]) {
                                s++;
                                break
                            }
                            n + 3 !== s - 2 && "slice_layer_without_partitioning_rbsp_idr" === Mi(31 & i[n + 3]) && (r = !0);
                            do {
                                s++
                            } while (1 !== i[s] && s < i.length);
                            n = s - 2,
                            s += 3;
                            break;
                        case 1:
                            if (0 !== i[s - 1] || 0 !== i[s - 2]) {
                                s += 3;
                                break
                            }
                            "slice_layer_without_partitioning_rbsp_idr" === Mi(31 & i[n + 3]) && (r = !0),
                            n = s - 2,
                            s += 3;
                            break;
                        default:
                            s += 3
                        }
                    return i = i.subarray(n),
                    s -= n,
                    n = 0,
                    i && i.byteLength > 3 && "slice_layer_without_partitioning_rbsp_idr" === Mi(31 & i[n + 3]) && (r = !0),
                    r
                }
            }
              , Ui = Ee
              , Bi = Le.handleRollover
              , Ni = {};
            Ni.ts = Ri,
            Ni.aac = gt;
            var Fi = ne.ONE_SECOND_IN_TS
              , ji = 188
              , $i = 71
              , qi = function(e, t, i) {
                for (var s, n, r, a, o = 0, l = ji, h = !1; l <= e.byteLength; )
                    if (e[o] !== $i || e[l] !== $i && l !== e.byteLength)
                        o++,
                        l++;
                    else {
                        switch (s = e.subarray(o, l),
                        Ni.ts.parseType(s, t.pid)) {
                        case "pes":
                            n = Ni.ts.parsePesType(s, t.table),
                            r = Ni.ts.parsePayloadUnitStartIndicator(s),
                            "audio" === n && r && (a = Ni.ts.parsePesTime(s)) && (a.type = "audio",
                            i.audio.push(a),
                            h = !0)
                        }
                        if (h)
                            break;
                        o += ji,
                        l += ji
                    }
                for (o = (l = e.byteLength) - ji,
                h = !1; o >= 0; )
                    if (e[o] !== $i || e[l] !== $i && l !== e.byteLength)
                        o--,
                        l--;
                    else {
                        switch (s = e.subarray(o, l),
                        Ni.ts.parseType(s, t.pid)) {
                        case "pes":
                            n = Ni.ts.parsePesType(s, t.table),
                            r = Ni.ts.parsePayloadUnitStartIndicator(s),
                            "audio" === n && r && (a = Ni.ts.parsePesTime(s)) && (a.type = "audio",
                            i.audio.push(a),
                            h = !0)
                        }
                        if (h)
                            break;
                        o -= ji,
                        l -= ji
                    }
            }
              , Hi = function(e, t, i) {
                for (var s, n, r, a, o, l, h, d = 0, u = ji, c = !1, p = {
                    data: [],
                    size: 0
                }; u < e.byteLength; )
                    if (e[d] !== $i || e[u] !== $i)
                        d++,
                        u++;
                    else {
                        switch (s = e.subarray(d, u),
                        Ni.ts.parseType(s, t.pid)) {
                        case "pes":
                            if (n = Ni.ts.parsePesType(s, t.table),
                            r = Ni.ts.parsePayloadUnitStartIndicator(s),
                            "video" === n && (r && !c && (a = Ni.ts.parsePesTime(s)) && (a.type = "video",
                            i.video.push(a),
                            c = !0),
                            !i.firstKeyFrame)) {
                                if (r && 0 !== p.size) {
                                    for (o = new Uint8Array(p.size),
                                    l = 0; p.data.length; )
                                        h = p.data.shift(),
                                        o.set(h, l),
                                        l += h.byteLength;
                                    if (Ni.ts.videoPacketContainsKeyFrame(o)) {
                                        var m = Ni.ts.parsePesTime(o);
                                        m ? (i.firstKeyFrame = m,
                                        i.firstKeyFrame.type = "video") : console.warn("Failed to extract PTS/DTS from PES at first keyframe. This could be an unusual TS segment, or else mux.js did not parse your TS segment correctly. If you know your TS segments do contain PTS/DTS on keyframes please file a bug report! You can try ffprobe to double check for yourself.")
                                    }
                                    p.size = 0
                                }
                                p.data.push(s),
                                p.size += s.byteLength
                            }
                        }
                        if (c && i.firstKeyFrame)
                            break;
                        d += ji,
                        u += ji
                    }
                for (d = (u = e.byteLength) - ji,
                c = !1; d >= 0; )
                    if (e[d] !== $i || e[u] !== $i)
                        d--,
                        u--;
                    else {
                        switch (s = e.subarray(d, u),
                        Ni.ts.parseType(s, t.pid)) {
                        case "pes":
                            n = Ni.ts.parsePesType(s, t.table),
                            r = Ni.ts.parsePayloadUnitStartIndicator(s),
                            "video" === n && r && (a = Ni.ts.parsePesTime(s)) && (a.type = "video",
                            i.video.push(a),
                            c = !0)
                        }
                        if (c)
                            break;
                        d -= ji,
                        u -= ji
                    }
            }
              , Vi = function(e) {
                var t = {
                    pid: null,
                    table: null
                }
                  , i = {};
                for (var s in function(e, t) {
                    for (var i, s = 0, n = ji; n < e.byteLength; )
                        if (e[s] !== $i || e[n] !== $i)
                            s++,
                            n++;
                        else {
                            switch (i = e.subarray(s, n),
                            Ni.ts.parseType(i, t.pid)) {
                            case "pat":
                                t.pid = Ni.ts.parsePat(i);
                                break;
                            case "pmt":
                                var r = Ni.ts.parsePmt(i);
                                t.table = t.table || {},
                                Object.keys(r).forEach((function(e) {
                                    t.table[e] = r[e]
                                }
                                ))
                            }
                            s += ji,
                            n += ji
                        }
                }(e, t),
                t.table) {
                    if (t.table.hasOwnProperty(s))
                        switch (t.table[s]) {
                        case Ui.H264_STREAM_TYPE:
                            i.video = [],
                            Hi(e, t, i),
                            0 === i.video.length && delete i.video;
                            break;
                        case Ui.ADTS_STREAM_TYPE:
                            i.audio = [],
                            qi(e, t, i),
                            0 === i.audio.length && delete i.audio
                        }
                }
                return i
            }
              , zi = function(e, t) {
                var i;
                return (i = Ni.aac.isLikelyAacData(e) ? function(e) {
                    for (var t, i = !1, s = 0, n = null, r = null, a = 0, o = 0; e.length - o >= 3; ) {
                        switch (Ni.aac.parseType(e, o)) {
                        case "timed-metadata":
                            if (e.length - o < 10) {
                                i = !0;
                                break
                            }
                            if ((a = Ni.aac.parseId3TagSize(e, o)) > e.length) {
                                i = !0;
                                break
                            }
                            null === r && (t = e.subarray(o, o + a),
                            r = Ni.aac.parseAacTimestamp(t)),
                            o += a;
                            break;
                        case "audio":
                            if (e.length - o < 7) {
                                i = !0;
                                break
                            }
                            if ((a = Ni.aac.parseAdtsSize(e, o)) > e.length) {
                                i = !0;
                                break
                            }
                            null === n && (t = e.subarray(o, o + a),
                            n = Ni.aac.parseSampleRate(t)),
                            s++,
                            o += a;
                            break;
                        default:
                            o++
                        }
                        if (i)
                            return null
                    }
                    if (null === n || null === r)
                        return null;
                    var l = Fi / n;
                    return {
                        audio: [{
                            type: "audio",
                            dts: r,
                            pts: r
                        }, {
                            type: "audio",
                            dts: r + 1024 * s * l,
                            pts: r + 1024 * s * l
                        }]
                    }
                }(e) : Vi(e)) && (i.audio || i.video) ? (function(e, t) {
                    if (e.audio && e.audio.length) {
                        var i = t;
                        ("undefined" === typeof i || isNaN(i)) && (i = e.audio[0].dts),
                        e.audio.forEach((function(e) {
                            e.dts = Bi(e.dts, i),
                            e.pts = Bi(e.pts, i),
                            e.dtsTime = e.dts / Fi,
                            e.ptsTime = e.pts / Fi
                        }
                        ))
                    }
                    if (e.video && e.video.length) {
                        var s = t;
                        if (("undefined" === typeof s || isNaN(s)) && (s = e.video[0].dts),
                        e.video.forEach((function(e) {
                            e.dts = Bi(e.dts, s),
                            e.pts = Bi(e.pts, s),
                            e.dtsTime = e.dts / Fi,
                            e.ptsTime = e.pts / Fi
                        }
                        )),
                        e.firstKeyFrame) {
                            var n = e.firstKeyFrame;
                            n.dts = Bi(n.dts, s),
                            n.pts = Bi(n.pts, s),
                            n.dtsTime = n.dts / Fi,
                            n.ptsTime = n.pts / Fi
                        }
                    }
                }(i, t),
                i) : null
            };
            class Wi {
                constructor(e, t) {
                    this.options = t || {},
                    this.self = e,
                    this.init()
                }
                init() {
                    this.transmuxer && this.transmuxer.dispose(),
                    this.transmuxer = new Vt.Transmuxer(this.options),
                    function(e, t) {
                        t.on("data", (function(t) {
                            const i = t.initSegment;
                            t.initSegment = {
                                data: i.buffer,
                                byteOffset: i.byteOffset,
                                byteLength: i.byteLength
                            };
                            const s = t.data;
                            t.data = s.buffer,
                            e.postMessage({
                                action: "data",
                                segment: t,
                                byteOffset: s.byteOffset,
                                byteLength: s.byteLength
                            }, [t.data])
                        }
                        )),
                        t.on("done", (function(t) {
                            e.postMessage({
                                action: "done"
                            })
                        }
                        )),
                        t.on("gopInfo", (function(t) {
                            e.postMessage({
                                action: "gopInfo",
                                gopInfo: t
                            })
                        }
                        )),
                        t.on("videoSegmentTimingInfo", (function(t) {
                            const i = {
                                start: {
                                    decode: ne.videoTsToSeconds(t.start.dts),
                                    presentation: ne.videoTsToSeconds(t.start.pts)
                                },
                                end: {
                                    decode: ne.videoTsToSeconds(t.end.dts),
                                    presentation: ne.videoTsToSeconds(t.end.pts)
                                },
                                baseMediaDecodeTime: ne.videoTsToSeconds(t.baseMediaDecodeTime)
                            };
                            t.prependedContentDuration && (i.prependedContentDuration = ne.videoTsToSeconds(t.prependedContentDuration)),
                            e.postMessage({
                                action: "videoSegmentTimingInfo",
                                videoSegmentTimingInfo: i
                            })
                        }
                        )),
                        t.on("audioSegmentTimingInfo", (function(t) {
                            const i = {
                                start: {
                                    decode: ne.videoTsToSeconds(t.start.dts),
                                    presentation: ne.videoTsToSeconds(t.start.pts)
                                },
                                end: {
                                    decode: ne.videoTsToSeconds(t.end.dts),
                                    presentation: ne.videoTsToSeconds(t.end.pts)
                                },
                                baseMediaDecodeTime: ne.videoTsToSeconds(t.baseMediaDecodeTime)
                            };
                            t.prependedContentDuration && (i.prependedContentDuration = ne.videoTsToSeconds(t.prependedContentDuration)),
                            e.postMessage({
                                action: "audioSegmentTimingInfo",
                                audioSegmentTimingInfo: i
                            })
                        }
                        )),
                        t.on("id3Frame", (function(t) {
                            e.postMessage({
                                action: "id3Frame",
                                id3Frame: t
                            })
                        }
                        )),
                        t.on("caption", (function(t) {
                            e.postMessage({
                                action: "caption",
                                caption: t
                            })
                        }
                        )),
                        t.on("trackinfo", (function(t) {
                            e.postMessage({
                                action: "trackinfo",
                                trackInfo: t
                            })
                        }
                        )),
                        t.on("audioTimingInfo", (function(t) {
                            e.postMessage({
                                action: "audioTimingInfo",
                                audioTimingInfo: {
                                    start: ne.videoTsToSeconds(t.start),
                                    end: ne.videoTsToSeconds(t.end)
                                }
                            })
                        }
                        )),
                        t.on("videoTimingInfo", (function(t) {
                            e.postMessage({
                                action: "videoTimingInfo",
                                videoTimingInfo: {
                                    start: ne.videoTsToSeconds(t.start),
                                    end: ne.videoTsToSeconds(t.end)
                                }
                            })
                        }
                        )),
                        t.on("log", (function(t) {
                            e.postMessage({
                                action: "log",
                                log: t
                            })
                        }
                        ))
                    }(this.self, this.transmuxer)
                }
                pushMp4Captions(e) {
                    this.captionParser || (this.captionParser = new ui,
                    this.captionParser.init());
                    const t = new Uint8Array(e.data,e.byteOffset,e.byteLength)
                      , i = this.captionParser.parse(t, e.trackIds, e.timescales);
                    this.self.postMessage({
                        action: "mp4Captions",
                        captions: i && i.captions || [],
                        logs: i && i.logs || [],
                        data: t.buffer
                    }, [t.buffer])
                }
                probeMp4StartTime({timescales: e, data: t}) {
                    const i = xi(e, t);
                    this.self.postMessage({
                        action: "probeMp4StartTime",
                        startTime: i,
                        data: t
                    }, [t.buffer])
                }
                probeMp4Tracks({data: e}) {
                    const t = Ii(e);
                    this.self.postMessage({
                        action: "probeMp4Tracks",
                        tracks: t,
                        data: e
                    }, [e.buffer])
                }
                probeEmsgID3({data: e, offset: t}) {
                    const i = Pi(e, t);
                    this.self.postMessage({
                        action: "probeEmsgID3",
                        id3Frames: i,
                        emsgData: e
                    }, [e.buffer])
                }
                probeTs({data: e, baseStartTime: t}) {
                    const i = "number" !== typeof t || isNaN(t) ? void 0 : t * ne.ONE_SECOND_IN_TS
                      , s = zi(e, i);
                    let n = null;
                    s && (n = {
                        hasVideo: s.video && 2 === s.video.length || !1,
                        hasAudio: s.audio && 2 === s.audio.length || !1
                    },
                    n.hasVideo && (n.videoStart = s.video[0].ptsTime),
                    n.hasAudio && (n.audioStart = s.audio[0].ptsTime)),
                    this.self.postMessage({
                        action: "probeTs",
                        result: n,
                        data: e
                    }, [e.buffer])
                }
                clearAllMp4Captions() {
                    this.captionParser && this.captionParser.clearAllCaptions()
                }
                clearParsedMp4Captions() {
                    this.captionParser && this.captionParser.clearParsedCaptions()
                }
                push(e) {
                    const t = new Uint8Array(e.data,e.byteOffset,e.byteLength);
                    this.transmuxer.push(t)
                }
                reset() {
                    this.transmuxer.reset()
                }
                setTimestampOffset(e) {
                    const t = e.timestampOffset || 0;
                    this.transmuxer.setBaseMediaDecodeTime(Math.round(ne.secondsToVideoTs(t)))
                }
                setAudioAppendStart(e) {
                    this.transmuxer.setAudioAppendStart(Math.ceil(ne.secondsToVideoTs(e.appendStart)))
                }
                setRemux(e) {
                    this.transmuxer.setRemux(e.remux)
                }
                flush(e) {
                    this.transmuxer.flush(),
                    self.postMessage({
                        action: "done",
                        type: "transmuxed"
                    })
                }
                endTimeline() {
                    this.transmuxer.endTimeline(),
                    self.postMessage({
                        action: "endedtimeline",
                        type: "transmuxed"
                    })
                }
                alignGopsWith(e) {
                    this.transmuxer.alignGopsWith(e.gopsToAlignWith.slice())
                }
            }
            self.onmessage = function(e) {
                "init" === e.data.action && e.data.options ? this.messageHandlers = new Wi(self,e.data.options) : (this.messageHandlers || (this.messageHandlers = new Wi(self)),
                e.data && e.data.action && "init" !== e.data.action && this.messageHandlers[e.data.action] && this.messageHandlers[e.data.action](e.data))
            }
        }
        ))));
        const Yr = e=>{
            const {transmuxer: t, bytes: i, audioAppendStart: s, gopsToAlignWith: n, remux: r, onData: a, onTrackInfo: o, onAudioTimingInfo: l, onVideoTimingInfo: h, onVideoSegmentTimingInfo: d, onAudioSegmentTimingInfo: u, onId3: c, onCaptions: p, onDone: m, onEndedTimeline: g, onTransmuxerLog: f, isEndOfTimeline: _} = e
              , y = {
                buffer: []
            };
            let v = _;
            if (t.onmessage = i=>{
                t.currentTransmux === e && ("data" === i.data.action && ((e,t,i)=>{
                    const {type: s, initSegment: n, captions: r, captionStreams: a, metadata: o, videoFrameDtsTime: l, videoFramePtsTime: h} = e.data.segment;
                    t.buffer.push({
                        captions: r,
                        captionStreams: a,
                        metadata: o
                    });
                    const d = e.data.segment.boxes || {
                        data: e.data.segment.data
                    }
                      , u = {
                        type: s,
                        data: new Uint8Array(d.data,d.data.byteOffset,d.data.byteLength),
                        initSegment: new Uint8Array(n.data,n.byteOffset,n.byteLength)
                    };
                    "undefined" !== typeof l && (u.videoFrameDtsTime = l),
                    "undefined" !== typeof h && (u.videoFramePtsTime = h),
                    i(u)
                }
                )(i, y, a),
                "trackinfo" === i.data.action && o(i.data.trackInfo),
                "gopInfo" === i.data.action && ((e,t)=>{
                    t.gopInfo = e.data.gopInfo
                }
                )(i, y),
                "audioTimingInfo" === i.data.action && l(i.data.audioTimingInfo),
                "videoTimingInfo" === i.data.action && h(i.data.videoTimingInfo),
                "videoSegmentTimingInfo" === i.data.action && d(i.data.videoSegmentTimingInfo),
                "audioSegmentTimingInfo" === i.data.action && u(i.data.audioSegmentTimingInfo),
                "id3Frame" === i.data.action && c([i.data.id3Frame], i.data.id3Frame.dispatchType),
                "caption" === i.data.action && p(i.data.caption),
                "endedtimeline" === i.data.action && (v = !1,
                g()),
                "log" === i.data.action && f(i.data.log),
                "transmuxed" === i.data.type && (v || (t.onmessage = null,
                (({transmuxedData: e, callback: t})=>{
                    e.buffer = [],
                    t(e)
                }
                )({
                    transmuxedData: y,
                    callback: m
                }),
                Jr(t))))
            }
            ,
            s && t.postMessage({
                action: "setAudioAppendStart",
                appendStart: s
            }),
            Array.isArray(n) && t.postMessage({
                action: "alignGopsWith",
                gopsToAlignWith: n
            }),
            "undefined" !== typeof r && t.postMessage({
                action: "setRemux",
                remux: r
            }),
            i.byteLength) {
                const e = i instanceof ArrayBuffer ? i : i.buffer
                  , s = i instanceof ArrayBuffer ? 0 : i.byteOffset;
                t.postMessage({
                    action: "push",
                    data: e,
                    byteOffset: s,
                    byteLength: i.byteLength
                }, [e])
            }
            _ && t.postMessage({
                action: "endTimeline"
            }),
            t.postMessage({
                action: "flush"
            })
        }
          , Jr = e=>{
            e.currentTransmux = null,
            e.transmuxQueue.length && (e.currentTransmux = e.transmuxQueue.shift(),
            "function" === typeof e.currentTransmux ? e.currentTransmux() : Yr(e.currentTransmux))
        }
          , Zr = (e,t)=>{
            e.postMessage({
                action: t
            }),
            Jr(e)
        }
          , ea = (e,t)=>{
            if (!t.currentTransmux)
                return t.currentTransmux = e,
                void Zr(t, e);
            t.transmuxQueue.push(Zr.bind(null, t, e))
        }
          , ta = e=>{
            if (!e.transmuxer.currentTransmux)
                return e.transmuxer.currentTransmux = e,
                void Yr(e);
            e.transmuxer.transmuxQueue.push(e)
        }
        ;
        var ia = e=>{
            ea("reset", e)
        }
          , sa = e=>{
            const t = new Xr;
            t.currentTransmux = null,
            t.transmuxQueue = [];
            const i = t.terminate;
            return t.terminate = ()=>(t.currentTransmux = null,
            t.transmuxQueue.length = 0,
            i.call(t)),
            t.postMessage({
                action: "init",
                options: e
            }),
            t
        }
        ;
        const na = function(e) {
            const t = e.transmuxer
              , i = e.endAction || e.action
              , s = e.callback
              , n = (0,
            g.Z)({}, e, {
                endAction: null,
                transmuxer: null,
                callback: null
            })
              , r = n=>{
                n.data.action === i && (t.removeEventListener("message", r),
                n.data.data && (n.data.data = new Uint8Array(n.data.data,e.byteOffset || 0,e.byteLength || n.data.data.byteLength),
                e.data && (e.data = n.data.data)),
                s(n.data))
            }
            ;
            if (t.addEventListener("message", r),
            e.data) {
                const i = e.data instanceof ArrayBuffer;
                n.byteOffset = i ? 0 : e.data.byteOffset,
                n.byteLength = e.data.byteLength;
                const s = [i ? e.data : e.data.buffer];
                t.postMessage(n, s)
            } else
                t.postMessage(n)
        }
          , ra = 2
          , aa = -101
          , oa = -102
          , la = e=>{
            e.forEach((e=>{
                e.abort()
            }
            ))
        }
          , ha = (e,t)=>t.timedout ? {
            status: t.status,
            message: "HLS request timed-out at URL: " + t.uri,
            code: aa,
            xhr: t
        } : t.aborted ? {
            status: t.status,
            message: "HLS request aborted at URL: " + t.uri,
            code: oa,
            xhr: t
        } : e ? {
            status: t.status,
            message: "HLS request errored at URL: " + t.uri,
            code: ra,
            xhr: t
        } : "arraybuffer" === t.responseType && 0 === t.response.byteLength ? {
            status: t.status,
            message: "Empty HLS response at URL: " + t.uri,
            code: ra,
            xhr: t
        } : null
          , da = (e,t,i)=>(s,n)=>{
            const r = n.response
              , a = ha(s, n);
            if (a)
                return i(a, e);
            if (16 !== r.byteLength)
                return i({
                    status: n.status,
                    message: "Invalid HLS key at URL: " + n.uri,
                    code: ra,
                    xhr: n
                }, e);
            const o = new DataView(r)
              , l = new Uint32Array([o.getUint32(0), o.getUint32(4), o.getUint32(8), o.getUint32(12)]);
            for (let e = 0; e < t.length; e++)
                t[e].bytes = l;
            return i(null, e)
        }
          , ua = (e,t)=>{
            const i = (0,
            E.Xm)(e.map.bytes);
            if ("mp4" !== i) {
                const s = e.map.resolvedUri || e.map.uri;
                return t({
                    internal: !0,
                    message: `Found unsupported ${i || "unknown"} container for initialization segment at URL: ${s}`,
                    code: ra
                })
            }
            na({
                action: "probeMp4Tracks",
                data: e.map.bytes,
                transmuxer: e.transmuxer,
                callback: ({tracks: i, data: s})=>(e.map.bytes = s,
                i.forEach((function(t) {
                    e.map.tracks = e.map.tracks || {},
                    e.map.tracks[t.type] || (e.map.tracks[t.type] = t,
                    "number" === typeof t.id && t.timescale && (e.map.timescales = e.map.timescales || {},
                    e.map.timescales[t.id] = t.timescale))
                }
                )),
                t(null))
            })
        }
          , ca = ({segment: e, finishProcessingFn: t, responseType: i})=>(s,n)=>{
            const r = ha(s, n);
            if (r)
                return t(r, e);
            const a = "arraybuffer" !== i && n.responseText ? (e=>{
                const t = new Uint8Array(new ArrayBuffer(e.length));
                for (let i = 0; i < e.length; i++)
                    t[i] = e.charCodeAt(i);
                return t.buffer
            }
            )(n.responseText.substring(e.lastReachedChar || 0)) : n.response;
            return e.stats = (e=>({
                bandwidth: e.bandwidth,
                bytesReceived: e.bytesReceived || 0,
                roundTripTime: e.roundTripTime || 0
            }))(n),
            e.key ? e.encryptedBytes = new Uint8Array(a) : e.bytes = new Uint8Array(a),
            t(null, e)
        }
          , pa = ({segment: e, bytes: t, trackInfoFn: i, timingInfoFn: s, videoSegmentTimingInfoFn: n, audioSegmentTimingInfoFn: r, id3Fn: a, captionsFn: o, isEndOfTimeline: l, endedTimelineFn: h, dataFn: d, doneFn: u, onTransmuxerLog: c})=>{
            const p = e.map && e.map.tracks || {}
              , m = Boolean(p.audio && p.video);
            let g = s.bind(null, e, "audio", "start");
            const f = s.bind(null, e, "audio", "end");
            let _ = s.bind(null, e, "video", "start");
            const y = s.bind(null, e, "video", "end");
            na({
                action: "probeTs",
                transmuxer: e.transmuxer,
                data: t,
                baseStartTime: e.baseStartTime,
                callback: s=>{
                    e.bytes = t = s.data;
                    const p = s.result;
                    p && (i(e, {
                        hasAudio: p.hasAudio,
                        hasVideo: p.hasVideo,
                        isMuxed: m
                    }),
                    i = null),
                    ta({
                        bytes: t,
                        transmuxer: e.transmuxer,
                        audioAppendStart: e.audioAppendStart,
                        gopsToAlignWith: e.gopsToAlignWith,
                        remux: m,
                        onData: t=>{
                            t.type = "combined" === t.type ? "video" : t.type,
                            d(e, t)
                        }
                        ,
                        onTrackInfo: t=>{
                            i && (m && (t.isMuxed = !0),
                            i(e, t))
                        }
                        ,
                        onAudioTimingInfo: e=>{
                            g && "undefined" !== typeof e.start && (g(e.start),
                            g = null),
                            f && "undefined" !== typeof e.end && f(e.end)
                        }
                        ,
                        onVideoTimingInfo: e=>{
                            _ && "undefined" !== typeof e.start && (_(e.start),
                            _ = null),
                            y && "undefined" !== typeof e.end && y(e.end)
                        }
                        ,
                        onVideoSegmentTimingInfo: e=>{
                            n(e)
                        }
                        ,
                        onAudioSegmentTimingInfo: e=>{
                            r(e)
                        }
                        ,
                        onId3: (t,i)=>{
                            a(e, t, i)
                        }
                        ,
                        onCaptions: t=>{
                            o(e, [t])
                        }
                        ,
                        isEndOfTimeline: l,
                        onEndedTimeline: ()=>{
                            h()
                        }
                        ,
                        onTransmuxerLog: c,
                        onDone: t=>{
                            u && (t.type = "combined" === t.type ? "video" : t.type,
                            u(null, e, t))
                        }
                    })
                }
            })
        }
          , ma = ({segment: e, bytes: t, trackInfoFn: i, timingInfoFn: s, videoSegmentTimingInfoFn: n, audioSegmentTimingInfoFn: r, id3Fn: a, captionsFn: o, isEndOfTimeline: l, endedTimelineFn: h, dataFn: d, doneFn: u, onTransmuxerLog: c})=>{
            let p = new Uint8Array(t);
            if ((0,
            E.cz)(p)) {
                e.isFmp4 = !0;
                const {tracks: n} = e.map
                  , r = {
                    isFmp4: !0,
                    hasVideo: !!n.video,
                    hasAudio: !!n.audio
                };
                n.audio && n.audio.codec && "enca" !== n.audio.codec && (r.audioCodec = n.audio.codec),
                n.video && n.video.codec && "encv" !== n.video.codec && (r.videoCodec = n.video.codec),
                n.video && n.audio && (r.isMuxed = !0),
                i(e, r);
                const l = (t,i)=>{
                    d(e, {
                        data: p,
                        type: r.hasAudio && !r.isMuxed ? "audio" : "video"
                    }),
                    i && i.length && a(e, i),
                    t && t.length && o(e, t),
                    u(null, e, {})
                }
                ;
                na({
                    action: "probeMp4StartTime",
                    timescales: e.map.timescales,
                    data: p,
                    transmuxer: e.transmuxer,
                    callback: ({data: i, startTime: a})=>{
                        t = i.buffer,
                        e.bytes = p = i,
                        r.hasAudio && !r.isMuxed && s(e, "audio", "start", a),
                        r.hasVideo && s(e, "video", "start", a),
                        na({
                            action: "probeEmsgID3",
                            data: p,
                            transmuxer: e.transmuxer,
                            offset: a,
                            callback: ({emsgData: i, id3Frames: s})=>{
                                t = i.buffer,
                                e.bytes = p = i,
                                n.video && i.byteLength && e.transmuxer ? na({
                                    action: "pushMp4Captions",
                                    endAction: "mp4Captions",
                                    transmuxer: e.transmuxer,
                                    data: p,
                                    timescales: e.map.timescales,
                                    trackIds: [n.video.id],
                                    callback: i=>{
                                        t = i.data.buffer,
                                        e.bytes = p = i.data,
                                        i.logs.forEach((function(e) {
                                            c(Mn(e, {
                                                stream: "mp4CaptionParser"
                                            }))
                                        }
                                        )),
                                        l(i.captions, s)
                                    }
                                }) : l(void 0, s)
                            }
                        })
                    }
                })
            } else if (e.transmuxer) {
                if ("undefined" === typeof e.container && (e.container = (0,
                E.Xm)(p)),
                "ts" !== e.container && "aac" !== e.container)
                    return i(e, {
                        hasAudio: !1,
                        hasVideo: !1
                    }),
                    void u(null, e, {});
                pa({
                    segment: e,
                    bytes: t,
                    trackInfoFn: i,
                    timingInfoFn: s,
                    videoSegmentTimingInfoFn: n,
                    audioSegmentTimingInfoFn: r,
                    id3Fn: a,
                    captionsFn: o,
                    isEndOfTimeline: l,
                    endedTimelineFn: h,
                    dataFn: d,
                    doneFn: u,
                    onTransmuxerLog: c
                })
            } else
                u(null, e, {})
        }
          , ga = function({id: e, key: t, encryptedBytes: i, decryptionWorker: s}, n) {
            const r = t=>{
                if (t.data.source === e) {
                    s.removeEventListener("message", r);
                    const e = t.data.decrypted;
                    n(new Uint8Array(e.bytes,e.byteOffset,e.byteLength))
                }
            }
            ;
            let a;
            s.addEventListener("message", r),
            a = t.bytes.slice ? t.bytes.slice() : new Uint32Array(Array.prototype.slice.call(t.bytes)),
            s.postMessage(Lr({
                source: e,
                encrypted: i,
                key: a,
                iv: t.iv
            }), [i.buffer, a.buffer])
        }
          , fa = ({activeXhrs: e, decryptionWorker: t, trackInfoFn: i, timingInfoFn: s, videoSegmentTimingInfoFn: n, audioSegmentTimingInfoFn: r, id3Fn: a, captionsFn: o, isEndOfTimeline: l, endedTimelineFn: h, dataFn: d, doneFn: u, onTransmuxerLog: c})=>{
            let p = 0
              , m = !1;
            return (g,f)=>{
                if (!m) {
                    if (g)
                        return m = !0,
                        la(e),
                        u(g, f);
                    if (p += 1,
                    p === e.length) {
                        const p = function() {
                            if (f.encryptedBytes)
                                return (({decryptionWorker: e, segment: t, trackInfoFn: i, timingInfoFn: s, videoSegmentTimingInfoFn: n, audioSegmentTimingInfoFn: r, id3Fn: a, captionsFn: o, isEndOfTimeline: l, endedTimelineFn: h, dataFn: d, doneFn: u, onTransmuxerLog: c})=>{
                                    ga({
                                        id: t.requestId,
                                        key: t.key,
                                        encryptedBytes: t.encryptedBytes,
                                        decryptionWorker: e
                                    }, (e=>{
                                        t.bytes = e,
                                        ma({
                                            segment: t,
                                            bytes: t.bytes,
                                            trackInfoFn: i,
                                            timingInfoFn: s,
                                            videoSegmentTimingInfoFn: n,
                                            audioSegmentTimingInfoFn: r,
                                            id3Fn: a,
                                            captionsFn: o,
                                            isEndOfTimeline: l,
                                            endedTimelineFn: h,
                                            dataFn: d,
                                            doneFn: u,
                                            onTransmuxerLog: c
                                        })
                                    }
                                    ))
                                }
                                )({
                                    decryptionWorker: t,
                                    segment: f,
                                    trackInfoFn: i,
                                    timingInfoFn: s,
                                    videoSegmentTimingInfoFn: n,
                                    audioSegmentTimingInfoFn: r,
                                    id3Fn: a,
                                    captionsFn: o,
                                    isEndOfTimeline: l,
                                    endedTimelineFn: h,
                                    dataFn: d,
                                    doneFn: u,
                                    onTransmuxerLog: c
                                });
                            ma({
                                segment: f,
                                bytes: f.bytes,
                                trackInfoFn: i,
                                timingInfoFn: s,
                                videoSegmentTimingInfoFn: n,
                                audioSegmentTimingInfoFn: r,
                                id3Fn: a,
                                captionsFn: o,
                                isEndOfTimeline: l,
                                endedTimelineFn: h,
                                dataFn: d,
                                doneFn: u,
                                onTransmuxerLog: c
                            })
                        };
                        if (f.endOfAllRequests = Date.now(),
                        f.map && f.map.encryptedBytes && !f.map.bytes)
                            return ga({
                                decryptionWorker: t,
                                id: f.requestId + "-init",
                                encryptedBytes: f.map.encryptedBytes,
                                key: f.map.key
                            }, (t=>{
                                f.map.bytes = t,
                                ua(f, (t=>{
                                    if (t)
                                        return la(e),
                                        u(t, f);
                                    p()
                                }
                                ))
                            }
                            ));
                        p()
                    }
                }
            }
        }
          , _a = ({segment: e, progressFn: t, trackInfoFn: i, timingInfoFn: s, videoSegmentTimingInfoFn: n, audioSegmentTimingInfoFn: r, id3Fn: a, captionsFn: o, isEndOfTimeline: l, endedTimelineFn: h, dataFn: d})=>i=>{
            if (!i.target.aborted)
                return e.stats = Mn(e.stats, (e=>{
                    const t = e.target
                      , i = {
                        bandwidth: 1 / 0,
                        bytesReceived: 0,
                        roundTripTime: Date.now() - t.requestTime || 0
                    };
                    return i.bytesReceived = e.loaded,
                    i.bandwidth = Math.floor(i.bytesReceived / i.roundTripTime * 8 * 1e3),
                    i
                }
                )(i)),
                !e.stats.firstBytesReceivedAt && e.stats.bytesReceived && (e.stats.firstBytesReceivedAt = Date.now()),
                t(i, e)
        }
          , ya = ({xhr: e, xhrOptions: t, decryptionWorker: i, segment: s, abortFn: n, progressFn: r, trackInfoFn: a, timingInfoFn: o, videoSegmentTimingInfoFn: l, audioSegmentTimingInfoFn: h, id3Fn: d, captionsFn: u, isEndOfTimeline: c, endedTimelineFn: p, dataFn: m, doneFn: g, onTransmuxerLog: f})=>{
            const _ = []
              , y = fa({
                activeXhrs: _,
                decryptionWorker: i,
                trackInfoFn: a,
                timingInfoFn: o,
                videoSegmentTimingInfoFn: l,
                audioSegmentTimingInfoFn: h,
                id3Fn: d,
                captionsFn: u,
                isEndOfTimeline: c,
                endedTimelineFn: p,
                dataFn: m,
                doneFn: g,
                onTransmuxerLog: f
            });
            if (s.key && !s.key.bytes) {
                const i = [s.key];
                s.map && !s.map.bytes && s.map.key && s.map.key.resolvedUri === s.key.resolvedUri && i.push(s.map.key);
                const n = e(Mn(t, {
                    uri: s.key.resolvedUri,
                    responseType: "arraybuffer"
                }), da(s, i, y));
                _.push(n)
            }
            if (s.map && !s.map.bytes) {
                if (s.map.key && (!s.key || s.key.resolvedUri !== s.map.key.resolvedUri)) {
                    const i = e(Mn(t, {
                        uri: s.map.key.resolvedUri,
                        responseType: "arraybuffer"
                    }), da(s, [s.map.key], y));
                    _.push(i)
                }
                const i = e(Mn(t, {
                    uri: s.map.resolvedUri,
                    responseType: "arraybuffer",
                    headers: xr(s.map)
                }), (({segment: e, finishProcessingFn: t})=>(i,s)=>{
                    const n = ha(i, s);
                    if (n)
                        return t(n, e);
                    const r = new Uint8Array(s.response);
                    if (e.map.key)
                        return e.map.encryptedBytes = r,
                        t(null, e);
                    e.map.bytes = r,
                    ua(e, (function(i) {
                        if (i)
                            return i.xhr = s,
                            i.status = s.status,
                            t(i, e);
                        t(null, e)
                    }
                    ))
                }
                )({
                    segment: s,
                    finishProcessingFn: y
                }));
                _.push(i)
            }
            const v = Mn(t, {
                uri: s.part && s.part.resolvedUri || s.resolvedUri,
                responseType: "arraybuffer",
                headers: xr(s)
            })
              , T = e(v, ca({
                segment: s,
                finishProcessingFn: y,
                responseType: v.responseType
            }));
            T.addEventListener("progress", _a({
                segment: s,
                progressFn: r,
                trackInfoFn: a,
                timingInfoFn: o,
                videoSegmentTimingInfoFn: l,
                audioSegmentTimingInfoFn: h,
                id3Fn: d,
                captionsFn: u,
                isEndOfTimeline: c,
                endedTimelineFn: p,
                dataFn: m
            })),
            _.push(T);
            const b = {};
            return _.forEach((e=>{
                e.addEventListener("loadend", (({loadendState: e, abortFn: t})=>i=>{
                    i.target.aborted && t && !e.calledAbortFn && (t(),
                    e.calledAbortFn = !0)
                }
                )({
                    loadendState: b,
                    abortFn: n
                }))
            }
            )),
            ()=>la(_)
        }
          , va = On("CodecUtils")
          , Ta = (e,t)=>{
            const i = t.attributes || {};
            return e && e.mediaGroups && e.mediaGroups.AUDIO && i.AUDIO && e.mediaGroups.AUDIO[i.AUDIO]
        }
          , ba = function(e) {
            const t = {};
            return e.forEach((({mediaType: e, type: i, details: s})=>{
                t[e] = t[e] || [],
                t[e].push((0,
                y.ws)(`${i}${s}`))
            }
            )),
            Object.keys(t).forEach((function(e) {
                if (t[e].length > 1)
                    return va(`multiple ${e} codecs found as attributes: ${t[e].join(", ")}. Setting playlist codecs to null so that we wait for mux.js to probe segments for real codecs.`),
                    void (t[e] = null);
                t[e] = t[e][0]
            }
            )),
            t
        }
          , Sa = function(e) {
            let t = 0;
            return e.audio && t++,
            e.video && t++,
            t
        }
          , ka = function(e, t) {
            const i = t.attributes || {}
              , s = ba(function(e) {
                const t = e.attributes || {};
                if (t.CODECS)
                    return (0,
                    y.kS)(t.CODECS)
            }(t) || []);
            if (Ta(e, t) && !s.audio && !((e,t)=>{
                if (!Ta(e, t))
                    return !0;
                const i = t.attributes || {}
                  , s = e.mediaGroups.AUDIO[i.AUDIO];
                for (const n in s)
                    if (!s[n].uri && !s[n].playlists)
                        return !0;
                return !1
            }
            )(e, t)) {
                const t = ba((0,
                y.Jg)(e, i.AUDIO) || []);
                t.audio && (s.audio = t.audio)
            }
            return s
        }
          , Ca = On("PlaylistSelector")
          , Ea = function(e) {
            if (!e || !e.playlist)
                return;
            const t = e.playlist;
            return JSON.stringify({
                id: t.id,
                bandwidth: e.bandwidth,
                width: e.width,
                height: e.height,
                codecs: t.attributes && t.attributes.CODECS || ""
            })
        }
          , wa = function(e, t) {
            if (!e)
                return "";
            const i = n().getComputedStyle(e);
            return i ? i[t] : ""
        }
          , xa = function(e, t) {
            const i = e.slice();
            e.sort((function(e, s) {
                const n = t(e, s);
                return 0 === n ? i.indexOf(e) - i.indexOf(s) : n
            }
            ))
        }
          , Ia = function(e, t) {
            let i, s;
            return e.attributes.BANDWIDTH && (i = e.attributes.BANDWIDTH),
            i = i || n().Number.MAX_VALUE,
            t.attributes.BANDWIDTH && (s = t.attributes.BANDWIDTH),
            s = s || n().Number.MAX_VALUE,
            i - s
        };
        let Pa = function(e, t, i, s, r, a) {
            if (!e)
                return;
            const o = {
                bandwidth: t,
                width: i,
                height: s,
                limitRenditionByPlayerDimensions: r
            };
            let l = e.playlists;
            lr.isAudioOnly(e) && (l = a.getAudioTrackPlaylists_(),
            o.audioOnly = !0);
            let h = l.map((e=>{
                let t;
                const i = e.attributes && e.attributes.RESOLUTION && e.attributes.RESOLUTION.width
                  , s = e.attributes && e.attributes.RESOLUTION && e.attributes.RESOLUTION.height;
                return t = e.attributes && e.attributes.BANDWIDTH,
                t = t || n().Number.MAX_VALUE,
                {
                    bandwidth: t,
                    width: i,
                    height: s,
                    playlist: e
                }
            }
            ));
            xa(h, ((e,t)=>e.bandwidth - t.bandwidth)),
            h = h.filter((e=>!lr.isIncompatible(e.playlist)));
            let d = h.filter((e=>lr.isEnabled(e.playlist)));
            d.length || (d = h.filter((e=>!lr.isDisabled(e.playlist))));
            const u = d.filter((e=>e.bandwidth * zr.BANDWIDTH_VARIANCE < t));
            let c = u[u.length - 1];
            const p = u.filter((e=>e.bandwidth === c.bandwidth))[0];
            if (!1 === r) {
                const e = p || d[0] || h[0];
                if (e && e.playlist) {
                    let t = "sortedPlaylistReps";
                    return p && (t = "bandwidthBestRep"),
                    d[0] && (t = "enabledPlaylistReps"),
                    Ca(`choosing ${Ea(e)} using ${t} with options`, o),
                    e.playlist
                }
                return Ca("could not choose a playlist with options", o),
                null
            }
            const m = u.filter((e=>e.width && e.height));
            xa(m, ((e,t)=>e.width - t.width));
            const g = m.filter((e=>e.width === i && e.height === s));
            c = g[g.length - 1];
            const f = g.filter((e=>e.bandwidth === c.bandwidth))[0];
            let _, y, v, T;
            if (f || (_ = m.filter((e=>e.width > i || e.height > s)),
            y = _.filter((e=>e.width === _[0].width && e.height === _[0].height)),
            c = y[y.length - 1],
            v = y.filter((e=>e.bandwidth === c.bandwidth))[0]),
            a.leastPixelDiffSelector) {
                const e = m.map((e=>(e.pixelDiff = Math.abs(e.width - i) + Math.abs(e.height - s),
                e)));
                xa(e, ((e,t)=>e.pixelDiff === t.pixelDiff ? t.bandwidth - e.bandwidth : e.pixelDiff - t.pixelDiff)),
                T = e[0]
            }
            const b = T || v || f || p || d[0] || h[0];
            if (b && b.playlist) {
                let e = "sortedPlaylistReps";
                return T ? e = "leastPixelDiffRep" : v ? e = "resolutionPlusOneRep" : f ? e = "resolutionBestRep" : p ? e = "bandwidthBestRep" : d[0] && (e = "enabledPlaylistReps"),
                Ca(`choosing ${Ea(b)} using ${e} with options`, o),
                b.playlist
            }
            return Ca("could not choose a playlist with options", o),
            null
        };
        const Aa = function() {
            const e = this.useDevicePixelRatio && n().devicePixelRatio || 1;
            return Pa(this.playlists.main, this.systemBandwidth, parseInt(wa(this.tech_.el(), "width"), 10) * e, parseInt(wa(this.tech_.el(), "height"), 10) * e, this.limitRenditionByPlayerDimensions, this.playlistController_)
        }
          , La = ({inbandTextTracks: e, metadataArray: t, timestampOffset: i, videoDuration: s})=>{
            if (!t)
                return;
            const r = n().WebKitDataCue || n().VTTCue
              , a = e.metadataTrack_;
            if (!a)
                return;
            if (t.forEach((e=>{
                const t = e.cueTime + i;
                !("number" !== typeof t || n().isNaN(t) || t < 0) && t < 1 / 0 && e.frames && e.frames.length && e.frames.forEach((e=>{
                    const i = new r(t,t,e.value || e.url || e.data || "");
                    i.frame = e,
                    i.value = e,
                    function(e) {
                        Object.defineProperties(e.frame, {
                            id: {
                                get: ()=>(wn.log.warn("cue.frame.id is deprecated. Use cue.value.key instead."),
                                e.value.key)
                            },
                            value: {
                                get: ()=>(wn.log.warn("cue.frame.value is deprecated. Use cue.value.data instead."),
                                e.value.data)
                            },
                            privateData: {
                                get: ()=>(wn.log.warn("cue.frame.privateData is deprecated. Use cue.value.data instead."),
                                e.value.data)
                            }
                        })
                    }(i),
                    a.addCue(i)
                }
                ))
            }
            )),
            !a.cues || !a.cues.length)
                return;
            const o = a.cues
              , l = [];
            for (let n = 0; n < o.length; n++)
                o[n] && l.push(o[n]);
            const h = l.reduce(((e,t)=>{
                const i = e[t.startTime] || [];
                return i.push(t),
                e[t.startTime] = i,
                e
            }
            ), {})
              , d = Object.keys(h).sort(((e,t)=>Number(e) - Number(t)));
            d.forEach(((e,t)=>{
                const i = h[e]
                  , n = isFinite(s) ? s : e
                  , r = Number(d[t + 1]) || n;
                i.forEach((e=>{
                    e.endTime = r
                }
                ))
            }
            ))
        }
          , Da = {
            id: "ID",
            class: "CLASS",
            startDate: "START-DATE",
            duration: "DURATION",
            endDate: "END-DATE",
            endOnNext: "END-ON-NEXT",
            plannedDuration: "PLANNED-DURATION",
            scte35Out: "SCTE35-OUT",
            scte35In: "SCTE35-IN"
        }
          , Oa = new Set(["id", "class", "startDate", "duration", "endDate", "endOnNext", "startTime", "endTime", "processDateRange"])
          , Ma = (e,t,i)=>{
            e.metadataTrack_ || (e.metadataTrack_ = i.addRemoteTextTrack({
                kind: "metadata",
                label: "Timed Metadata"
            }, !1).track,
            wn.browser.IS_ANY_SAFARI || (e.metadataTrack_.inBandMetadataTrackDispatchType = t))
        }
          , Ra = function(e, t, i) {
            let s, n;
            if (i && i.cues)
                for (s = i.cues.length; s--; )
                    n = i.cues[s],
                    n.startTime >= e && n.endTime <= t && i.removeCue(n)
        }
          , Ua = e=>"number" === typeof e && isFinite(e)
          , Ba = 1 / 60
          , Na = e=>{
            const {startOfSegment: t, duration: i, segment: s, part: n, playlist: {mediaSequence: r, id: a, segments: o=[]}, mediaIndex: l, partIndex: h, timeline: d} = e
              , u = o.length - 1;
            let c = "mediaIndex/partIndex increment";
            e.getMediaInfoForTime ? c = `getMediaInfoForTime (${e.getMediaInfoForTime})` : e.isSyncRequest && (c = "getSyncSegmentCandidate (isSyncRequest)"),
            e.independent && (c += ` with independent ${e.independent}`);
            const p = "number" === typeof h
              , m = e.segment.uri ? "segment" : "pre-segment"
              , g = p ? Kn({
                preloadSegment: s
            }) - 1 : 0;
            return `${m} [${r + l}/${r + u}]` + (p ? ` part [${h}/${g}]` : "") + ` segment start/end [${s.start} => ${s.end}]` + (p ? ` part start/end [${n.start} => ${n.end}]` : "") + ` startOfSegment [${t}]` + ` duration [${i}]` + ` timeline [${d}]` + ` selected by [${c}]` + ` playlist [${a}]`
        }
          , Fa = e=>`${e}TimingInfo`
          , ja = ({timelineChangeController: e, currentTimeline: t, segmentTimeline: i, loaderType: s, audioDisabled: n})=>{
            if (t === i)
                return !1;
            if ("audio" === s) {
                const t = e.lastTimelineChange({
                    type: "main"
                });
                return !t || t.to !== i
            }
            if ("main" === s && n) {
                const t = e.pendingTimelineChange({
                    type: "audio"
                });
                return !t || t.to !== i
            }
            return !1
        }
          , $a = ({segmentDuration: e, maxDuration: t})=>!!e && Math.round(e) > t + Un
          , qa = (e,t)=>{
            if ("hls" !== t)
                return null;
            const i = (e=>{
                let t = 0;
                return ["video", "audio"].forEach((function(i) {
                    const s = e[`${i}TimingInfo`];
                    if (!s)
                        return;
                    const {start: r, end: a} = s;
                    let o;
                    "bigint" === typeof r || "bigint" === typeof a ? o = n().BigInt(a) - n().BigInt(r) : "number" === typeof r && "number" === typeof a && (o = a - r),
                    "undefined" !== typeof o && o > t && (t = o)
                }
                )),
                "bigint" === typeof t && t < Number.MAX_SAFE_INTEGER && (t = Number(t)),
                t
            }
            )({
                audioTimingInfo: e.audioTimingInfo,
                videoTimingInfo: e.videoTimingInfo
            });
            if (!i)
                return null;
            const s = e.playlist.targetDuration
              , r = $a({
                segmentDuration: i,
                maxDuration: 2 * s
            })
              , a = $a({
                segmentDuration: i,
                maxDuration: s
            })
              , o = `Segment with index ${e.mediaIndex} from playlist ${e.playlist.id} has a duration of ${i} when the reported duration is ${e.duration} and the target duration is ${s}. For HLS content, a duration in excess of the target duration may result in playback issues. See the HLS specification section on EXT-X-TARGETDURATION for more details: https://tools.ietf.org/html/draft-pantos-http-live-streaming-23#section-4.3.3.1`;
            return r || a ? {
                severity: r ? "warn" : "info",
                message: o
            } : null
        }
        ;
        class Ha extends wn.EventTarget {
            constructor(e, t={}) {
                if (super(),
                !e)
                    throw new TypeError("Initialization settings are required");
                if ("function" !== typeof e.currentTime)
                    throw new TypeError("No currentTime getter specified");
                if (!e.mediaSource)
                    throw new TypeError("No MediaSource specified");
                this.bandwidth = e.bandwidth,
                this.throughput = {
                    rate: 0,
                    count: 0
                },
                this.roundTrip = NaN,
                this.resetStats_(),
                this.mediaIndex = null,
                this.partIndex = null,
                this.hasPlayed_ = e.hasPlayed,
                this.currentTime_ = e.currentTime,
                this.seekable_ = e.seekable,
                this.seeking_ = e.seeking,
                this.duration_ = e.duration,
                this.mediaSource_ = e.mediaSource,
                this.vhs_ = e.vhs,
                this.loaderType_ = e.loaderType,
                this.currentMediaInfo_ = void 0,
                this.startingMediaInfo_ = void 0,
                this.segmentMetadataTrack_ = e.segmentMetadataTrack,
                this.goalBufferLength_ = e.goalBufferLength,
                this.sourceType_ = e.sourceType,
                this.sourceUpdater_ = e.sourceUpdater,
                this.inbandTextTracks_ = e.inbandTextTracks,
                this.state_ = "INIT",
                this.timelineChangeController_ = e.timelineChangeController,
                this.shouldSaveSegmentTimingInfo_ = !0,
                this.parse708captions_ = e.parse708captions,
                this.useDtsForTimestampOffset_ = e.useDtsForTimestampOffset,
                this.captionServices_ = e.captionServices,
                this.exactManifestTimings = e.exactManifestTimings,
                this.addMetadataToTextTrack = e.addMetadataToTextTrack,
                this.checkBufferTimeout_ = null,
                this.error_ = void 0,
                this.currentTimeline_ = -1,
                this.shouldForceTimestampOffsetAfterResync_ = !1,
                this.pendingSegment_ = null,
                this.xhrOptions_ = null,
                this.pendingSegments_ = [],
                this.audioDisabled_ = !1,
                this.isPendingTimestampOffset_ = !1,
                this.gopBuffer_ = [],
                this.timeMapping_ = 0,
                this.safeAppend_ = !1,
                this.appendInitSegment_ = {
                    audio: !0,
                    video: !0
                },
                this.playlistOfLastInitSegment_ = {
                    audio: null,
                    video: null
                },
                this.callQueue_ = [],
                this.loadQueue_ = [],
                this.metadataQueue_ = {
                    id3: [],
                    caption: []
                },
                this.waitingOnRemove_ = !1,
                this.quotaExceededErrorRetryTimeout_ = null,
                this.activeInitSegmentId_ = null,
                this.initSegments_ = {},
                this.cacheEncryptionKeys_ = e.cacheEncryptionKeys,
                this.keyCache_ = {},
                this.decrypter_ = e.decrypter,
                this.syncController_ = e.syncController,
                this.syncPoint_ = {
                    segmentIndex: 0,
                    time: 0
                },
                this.transmuxer_ = this.createTransmuxer_(),
                this.triggerSyncInfoUpdate_ = ()=>this.trigger("syncinfoupdate"),
                this.syncController_.on("syncinfoupdate", this.triggerSyncInfoUpdate_),
                this.mediaSource_.addEventListener("sourceopen", (()=>{
                    this.isEndOfStream_() || (this.ended_ = !1)
                }
                )),
                this.fetchAtBuffer_ = !1,
                this.logger_ = On(`SegmentLoader[${this.loaderType_}]`),
                Object.defineProperty(this, "state", {
                    get() {
                        return this.state_
                    },
                    set(e) {
                        e !== this.state_ && (this.logger_(`${this.state_} -> ${e}`),
                        this.state_ = e,
                        this.trigger("statechange"))
                    }
                }),
                this.sourceUpdater_.on("ready", (()=>{
                    this.hasEnoughInfoToAppend_() && this.processCallQueue_()
                }
                )),
                "main" === this.loaderType_ && this.timelineChangeController_.on("pendingtimelinechange", (()=>{
                    this.hasEnoughInfoToAppend_() && this.processCallQueue_()
                }
                )),
                "audio" === this.loaderType_ && this.timelineChangeController_.on("timelinechange", (()=>{
                    this.hasEnoughInfoToLoad_() && this.processLoadQueue_(),
                    this.hasEnoughInfoToAppend_() && this.processCallQueue_()
                }
                ))
            }
            createTransmuxer_() {
                return sa({
                    remux: !1,
                    alignGopsAtEnd: this.safeAppend_,
                    keepOriginalTimestamps: !0,
                    parse708captions: this.parse708captions_,
                    captionServices: this.captionServices_
                })
            }
            resetStats_() {
                this.mediaBytesTransferred = 0,
                this.mediaRequests = 0,
                this.mediaRequestsAborted = 0,
                this.mediaRequestsTimedout = 0,
                this.mediaRequestsErrored = 0,
                this.mediaTransferDuration = 0,
                this.mediaSecondsLoaded = 0,
                this.mediaAppends = 0
            }
            dispose() {
                this.trigger("dispose"),
                this.state = "DISPOSED",
                this.pause(),
                this.abort_(),
                this.transmuxer_ && this.transmuxer_.terminate(),
                this.resetStats_(),
                this.checkBufferTimeout_ && n().clearTimeout(this.checkBufferTimeout_),
                this.syncController_ && this.triggerSyncInfoUpdate_ && this.syncController_.off("syncinfoupdate", this.triggerSyncInfoUpdate_),
                this.off()
            }
            setAudio(e) {
                this.audioDisabled_ = !e,
                e ? this.appendInitSegment_.audio = !0 : this.sourceUpdater_.removeAudio(0, this.duration_())
            }
            abort() {
                "WAITING" === this.state ? (this.abort_(),
                this.state = "READY",
                this.paused() || this.monitorBuffer_()) : this.pendingSegment_ && (this.pendingSegment_ = null)
            }
            abort_() {
                this.pendingSegment_ && this.pendingSegment_.abortRequests && this.pendingSegment_.abortRequests(),
                this.pendingSegment_ = null,
                this.callQueue_ = [],
                this.loadQueue_ = [],
                this.metadataQueue_.id3 = [],
                this.metadataQueue_.caption = [],
                this.timelineChangeController_.clearPendingTimelineChange(this.loaderType_),
                this.waitingOnRemove_ = !1,
                n().clearTimeout(this.quotaExceededErrorRetryTimeout_),
                this.quotaExceededErrorRetryTimeout_ = null
            }
            checkForAbort_(e) {
                return "APPENDING" !== this.state || this.pendingSegment_ ? !this.pendingSegment_ || this.pendingSegment_.requestId !== e : (this.state = "READY",
                !0)
            }
            error(e) {
                return "undefined" !== typeof e && (this.logger_("error occurred:", e),
                this.error_ = e),
                this.pendingSegment_ = null,
                this.error_
            }
            endOfStream() {
                this.ended_ = !0,
                this.transmuxer_ && ia(this.transmuxer_),
                this.gopBuffer_.length = 0,
                this.pause(),
                this.trigger("ended")
            }
            buffered_() {
                const e = this.getMediaInfo_();
                if (!this.sourceUpdater_ || !e)
                    return Rn();
                if ("main" === this.loaderType_) {
                    const {hasAudio: t, hasVideo: i, isMuxed: s} = e;
                    if (i && t && !this.audioDisabled_ && !s)
                        return this.sourceUpdater_.buffered();
                    if (i)
                        return this.sourceUpdater_.videoBuffered()
                }
                return this.sourceUpdater_.audioBuffered()
            }
            initSegmentForMap(e, t=!1) {
                if (!e)
                    return null;
                const i = Dr(e);
                let s = this.initSegments_[i];
                return t && !s && e.bytes && (this.initSegments_[i] = s = {
                    resolvedUri: e.resolvedUri,
                    byterange: e.byterange,
                    bytes: e.bytes,
                    tracks: e.tracks,
                    timescales: e.timescales
                }),
                s || e
            }
            segmentKey(e, t=!1) {
                if (!e)
                    return null;
                const i = Or(e);
                let s = this.keyCache_[i];
                this.cacheEncryptionKeys_ && t && !s && e.bytes && (this.keyCache_[i] = s = {
                    resolvedUri: e.resolvedUri,
                    bytes: e.bytes
                });
                const n = {
                    resolvedUri: (s || e).resolvedUri
                };
                return s && (n.bytes = s.bytes),
                n
            }
            couldBeginLoading_() {
                return this.playlist_ && !this.paused()
            }
            load() {
                if (this.monitorBuffer_(),
                this.playlist_)
                    return "INIT" === this.state && this.couldBeginLoading_() ? this.init_() : void (!this.couldBeginLoading_() || "READY" !== this.state && "INIT" !== this.state || (this.state = "READY"))
            }
            init_() {
                return this.state = "READY",
                this.resetEverything(),
                this.monitorBuffer_()
            }
            playlist(e, t={}) {
                if (!e)
                    return;
                const i = this.playlist_
                  , s = this.pendingSegment_;
                this.playlist_ = e,
                this.xhrOptions_ = t,
                "INIT" === this.state && (e.syncInfo = {
                    mediaSequence: e.mediaSequence,
                    time: 0
                },
                "main" === this.loaderType_ && this.syncController_.setDateTimeMappingForStart(e));
                let n = null;
                if (i && (i.id ? n = i.id : i.uri && (n = i.uri)),
                this.logger_(`playlist update [${n} => ${e.id || e.uri}]`),
                this.syncController_.updateMediaSequenceMap(e, this.currentTime_(), this.loaderType_),
                this.trigger("syncinfoupdate"),
                "INIT" === this.state && this.couldBeginLoading_())
                    return this.init_();
                if (!i || i.uri !== e.uri) {
                    if (null !== this.mediaIndex) {
                        !e.endList && "number" === typeof e.partTargetDuration ? this.resetLoader() : this.resyncLoader()
                    }
                    return this.currentMediaInfo_ = void 0,
                    void this.trigger("playlistupdate")
                }
                const r = e.mediaSequence - i.mediaSequence;
                if (this.logger_(`live window shift [${r}]`),
                null !== this.mediaIndex)
                    if (this.mediaIndex -= r,
                    this.mediaIndex < 0)
                        this.mediaIndex = null,
                        this.partIndex = null;
                    else {
                        const e = this.playlist_.segments[this.mediaIndex];
                        if (this.partIndex && (!e.parts || !e.parts.length || !e.parts[this.partIndex])) {
                            const e = this.mediaIndex;
                            this.logger_(`currently processing part (index ${this.partIndex}) no longer exists.`),
                            this.resetLoader(),
                            this.mediaIndex = e
                        }
                    }
                s && (s.mediaIndex -= r,
                s.mediaIndex < 0 ? (s.mediaIndex = null,
                s.partIndex = null) : (s.mediaIndex >= 0 && (s.segment = e.segments[s.mediaIndex]),
                s.partIndex >= 0 && s.segment.parts && (s.part = s.segment.parts[s.partIndex]))),
                this.syncController_.saveExpiredSegmentInfo(i, e)
            }
            pause() {
                this.checkBufferTimeout_ && (n().clearTimeout(this.checkBufferTimeout_),
                this.checkBufferTimeout_ = null)
            }
            paused() {
                return null === this.checkBufferTimeout_
            }
            resetEverything(e) {
                this.ended_ = !1,
                this.activeInitSegmentId_ = null,
                this.appendInitSegment_ = {
                    audio: !0,
                    video: !0
                },
                this.resetLoader(),
                this.remove(0, 1 / 0, e),
                this.transmuxer_ && (this.transmuxer_.postMessage({
                    action: "clearAllMp4Captions"
                }),
                this.transmuxer_.postMessage({
                    action: "reset"
                }))
            }
            resetLoader() {
                this.fetchAtBuffer_ = !1,
                this.resyncLoader()
            }
            resyncLoader() {
                this.transmuxer_ && ia(this.transmuxer_),
                this.mediaIndex = null,
                this.partIndex = null,
                this.syncPoint_ = null,
                this.isPendingTimestampOffset_ = !1,
                this.shouldForceTimestampOffsetAfterResync_ = !0,
                this.callQueue_ = [],
                this.loadQueue_ = [],
                this.metadataQueue_.id3 = [],
                this.metadataQueue_.caption = [],
                this.abort(),
                this.transmuxer_ && this.transmuxer_.postMessage({
                    action: "clearParsedMp4Captions"
                })
            }
            remove(e, t, i=(()=>{}
            ), s=!1) {
                if (t === 1 / 0 && (t = this.duration_()),
                t <= e)
                    return void this.logger_("skipping remove because end ${end} is <= start ${start}");
                if (!this.sourceUpdater_ || !this.getMediaInfo_())
                    return void this.logger_("skipping remove because no source updater or starting media info");
                let n = 1;
                const r = ()=>{
                    n--,
                    0 === n && i()
                }
                ;
                !s && this.audioDisabled_ || (n++,
                this.sourceUpdater_.removeAudio(e, t, r)),
                (s || "main" === this.loaderType_) && (this.gopBuffer_ = ((e,t,i,s)=>{
                    const n = Math.ceil((t - s) * w.ONE_SECOND_IN_TS)
                      , r = Math.ceil((i - s) * w.ONE_SECOND_IN_TS)
                      , a = e.slice();
                    let o = e.length;
                    for (; o-- && !(e[o].pts <= r); )
                        ;
                    if (-1 === o)
                        return a;
                    let l = o + 1;
                    for (; l-- && !(e[l].pts <= n); )
                        ;
                    return l = Math.max(l, 0),
                    a.splice(l, o - l + 1),
                    a
                }
                )(this.gopBuffer_, e, t, this.timeMapping_),
                n++,
                this.sourceUpdater_.removeVideo(e, t, r));
                for (const a in this.inbandTextTracks_)
                    Ra(e, t, this.inbandTextTracks_[a]);
                Ra(e, t, this.segmentMetadataTrack_),
                r()
            }
            monitorBuffer_() {
                this.checkBufferTimeout_ && n().clearTimeout(this.checkBufferTimeout_),
                this.checkBufferTimeout_ = n().setTimeout(this.monitorBufferTick_.bind(this), 1)
            }
            monitorBufferTick_() {
                "READY" === this.state && this.fillBuffer_(),
                this.checkBufferTimeout_ && n().clearTimeout(this.checkBufferTimeout_),
                this.checkBufferTimeout_ = n().setTimeout(this.monitorBufferTick_.bind(this), 500)
            }
            fillBuffer_() {
                if (this.sourceUpdater_.updating())
                    return;
                const e = this.chooseNextRequest_();
                e && ("number" === typeof e.timestampOffset && (this.isPendingTimestampOffset_ = !1,
                this.timelineChangeController_.pendingTimelineChange({
                    type: this.loaderType_,
                    from: this.currentTimeline_,
                    to: e.timeline
                })),
                this.loadSegment_(e))
            }
            isEndOfStream_(e=this.mediaIndex, t=this.playlist_, i=this.partIndex) {
                if (!t || !this.mediaSource_)
                    return !1;
                const s = "number" === typeof e && t.segments[e]
                  , n = e + 1 === t.segments.length
                  , r = !s || !s.parts || i + 1 === s.parts.length;
                return t.endList && "open" === this.mediaSource_.readyState && n && r
            }
            chooseNextRequest_() {
                const e = this.buffered_()
                  , t = Hn(e) || 0
                  , i = Vn(e, this.currentTime_())
                  , s = !this.hasPlayed_() && i >= 1
                  , n = i >= this.goalBufferLength_()
                  , r = this.playlist_.segments;
                if (!r.length || s || n)
                    return null;
                this.syncPoint_ = this.syncPoint_ || this.syncController_.getSyncPoint(this.playlist_, this.duration_(), this.currentTimeline_, this.currentTime_(), this.loaderType_);
                const a = {
                    partIndex: null,
                    mediaIndex: null,
                    startOfSegment: null,
                    playlist: this.playlist_,
                    isSyncRequest: Boolean(!this.syncPoint_)
                };
                if (a.isSyncRequest)
                    a.mediaIndex = function(e, t, i) {
                        t = t || [];
                        const s = [];
                        let n = 0;
                        for (let r = 0; r < t.length; r++) {
                            const a = t[r];
                            if (e === a.timeline && (s.push(r),
                            n += a.duration,
                            n > i))
                                return r
                        }
                        return 0 === s.length ? 0 : s[s.length - 1]
                    }(this.currentTimeline_, r, t),
                    this.logger_(`choose next request. Can not find sync point. Fallback to media Index: ${a.mediaIndex}`);
                else if (null !== this.mediaIndex) {
                    const e = r[this.mediaIndex]
                      , i = "number" === typeof this.partIndex ? this.partIndex : -1;
                    a.startOfSegment = e.end ? e.end : t,
                    e.parts && e.parts[i + 1] ? (a.mediaIndex = this.mediaIndex,
                    a.partIndex = i + 1) : a.mediaIndex = this.mediaIndex + 1
                } else {
                    const {segmentIndex: e, startTime: i, partIndex: s} = lr.getMediaInfoForTime({
                        exactManifestTimings: this.exactManifestTimings,
                        playlist: this.playlist_,
                        currentTime: this.fetchAtBuffer_ ? t : this.currentTime_(),
                        startingPartIndex: this.syncPoint_.partIndex,
                        startingSegmentIndex: this.syncPoint_.segmentIndex,
                        startTime: this.syncPoint_.time
                    });
                    a.getMediaInfoForTime = this.fetchAtBuffer_ ? `bufferedEnd ${t}` : `currentTime ${this.currentTime_()}`,
                    a.mediaIndex = e,
                    a.startOfSegment = i,
                    a.partIndex = s,
                    this.logger_(`choose next request. Playlist switched and we have a sync point. Media Index: ${a.mediaIndex} `)
                }
                const o = r[a.mediaIndex];
                let l = o && "number" === typeof a.partIndex && o.parts && o.parts[a.partIndex];
                if (!o || "number" === typeof a.partIndex && !l)
                    return null;
                "number" !== typeof a.partIndex && o.parts && (a.partIndex = 0,
                l = o.parts[0]);
                const h = this.vhs_.playlists && this.vhs_.playlists.main && this.vhs_.playlists.main.independentSegments || this.playlist_.independentSegments;
                if (!i && l && !h && !l.independent)
                    if (0 === a.partIndex) {
                        const e = r[a.mediaIndex - 1]
                          , t = e.parts && e.parts.length && e.parts[e.parts.length - 1];
                        t && t.independent && (a.mediaIndex -= 1,
                        a.partIndex = e.parts.length - 1,
                        a.independent = "previous segment")
                    } else
                        o.parts[a.partIndex - 1].independent && (a.partIndex -= 1,
                        a.independent = "previous part");
                const d = this.mediaSource_ && "ended" === this.mediaSource_.readyState;
                return a.mediaIndex >= r.length - 1 && d && !this.seeking_() ? null : (this.shouldForceTimestampOffsetAfterResync_ && (this.shouldForceTimestampOffsetAfterResync_ = !1,
                a.forceTimestampOffset = !0,
                this.logger_("choose next request. Force timestamp offset after loader resync")),
                this.generateSegmentInfo_(a))
            }
            generateSegmentInfo_(e) {
                const {independent: t, playlist: i, mediaIndex: s, startOfSegment: n, isSyncRequest: r, partIndex: a, forceTimestampOffset: o, getMediaInfoForTime: l} = e
                  , h = i.segments[s]
                  , d = "number" === typeof a && h.parts[a]
                  , u = {
                    requestId: "segment-loader-" + Math.random(),
                    uri: d && d.resolvedUri || h.resolvedUri,
                    mediaIndex: s,
                    partIndex: d ? a : null,
                    isSyncRequest: r,
                    startOfSegment: n,
                    playlist: i,
                    bytes: null,
                    encryptedBytes: null,
                    timestampOffset: null,
                    timeline: h.timeline,
                    duration: d && d.duration || h.duration,
                    segment: h,
                    part: d,
                    byteLength: 0,
                    transmuxer: this.transmuxer_,
                    getMediaInfoForTime: l,
                    independent: t
                }
                  , c = "undefined" !== typeof o ? o : this.isPendingTimestampOffset_;
                u.timestampOffset = this.timestampOffsetForSegment_({
                    segmentTimeline: h.timeline,
                    currentTimeline: this.currentTimeline_,
                    startOfSegment: n,
                    buffered: this.buffered_(),
                    overrideCheck: c
                });
                const p = Hn(this.sourceUpdater_.audioBuffered());
                return "number" === typeof p && (u.audioAppendStart = p - this.sourceUpdater_.audioTimestampOffset()),
                this.sourceUpdater_.videoBuffered().length && (u.gopsToAlignWith = ((e,t,i)=>{
                    if ("undefined" === typeof t || null === t || !e.length)
                        return [];
                    const s = Math.ceil((t - i + 3) * w.ONE_SECOND_IN_TS);
                    let n;
                    for (n = 0; n < e.length && !(e[n].pts > s); n++)
                        ;
                    return e.slice(n)
                }
                )(this.gopBuffer_, this.currentTime_() - this.sourceUpdater_.videoTimestampOffset(), this.timeMapping_)),
                u
            }
            timestampOffsetForSegment_(e) {
                return (({segmentTimeline: e, currentTimeline: t, startOfSegment: i, buffered: s, overrideCheck: n})=>n || e !== t ? e < t ? i : s.length ? s.end(s.length - 1) : i : null)(e)
            }
            earlyAbortWhenNeeded_(e) {
                if (this.vhs_.tech_.paused() || !this.xhrOptions_.timeout || !this.playlist_.attributes.BANDWIDTH)
                    return;
                if (Date.now() - (e.firstBytesReceivedAt || Date.now()) < 1e3)
                    return;
                const t = this.currentTime_()
                  , i = e.bandwidth
                  , s = this.pendingSegment_.duration
                  , n = lr.estimateSegmentRequestTime(s, i, this.playlist_, e.bytesReceived)
                  , r = function(e, t, i=1) {
                    return ((e.length ? e.end(e.length - 1) : 0) - t) / i
                }(this.buffered_(), t, this.vhs_.tech_.playbackRate()) - 1;
                if (n <= r)
                    return;
                const a = function(e) {
                    const {main: t, currentTime: i, bandwidth: s, duration: n, segmentDuration: r, timeUntilRebuffer: a, currentTimeline: o, syncController: l} = e
                      , h = t.playlists.filter((e=>!lr.isIncompatible(e)));
                    let d = h.filter(lr.isEnabled);
                    d.length || (d = h.filter((e=>!lr.isDisabled(e))));
                    const u = d.filter(lr.hasAttribute.bind(null, "BANDWIDTH")).map((e=>{
                        const t = l.getSyncPoint(e, n, o, i) ? 1 : 2;
                        return {
                            playlist: e,
                            rebufferingImpact: lr.estimateSegmentRequestTime(r, s, e) * t - a
                        }
                    }
                    ))
                      , c = u.filter((e=>e.rebufferingImpact <= 0));
                    return xa(c, ((e,t)=>Ia(t.playlist, e.playlist))),
                    c.length ? c[0] : (xa(u, ((e,t)=>e.rebufferingImpact - t.rebufferingImpact)),
                    u[0] || null)
                }({
                    main: this.vhs_.playlists.main,
                    currentTime: t,
                    bandwidth: i,
                    duration: this.duration_(),
                    segmentDuration: s,
                    timeUntilRebuffer: r,
                    currentTimeline: this.currentTimeline_,
                    syncController: this.syncController_
                });
                if (!a)
                    return;
                const o = n - r - a.rebufferingImpact;
                let l = .5;
                r <= Un && (l = 1),
                !a.playlist || a.playlist.uri === this.playlist_.uri || o < l || (this.bandwidth = a.playlist.attributes.BANDWIDTH * zr.BANDWIDTH_VARIANCE + 1,
                this.trigger("earlyabort"))
            }
            handleAbort_(e) {
                this.logger_(`Aborting ${Na(e)}`),
                this.mediaRequestsAborted += 1
            }
            handleProgress_(e, t) {
                this.earlyAbortWhenNeeded_(t.stats),
                this.checkForAbort_(t.requestId) || this.trigger("progress")
            }
            handleTrackInfo_(e, t) {
                this.earlyAbortWhenNeeded_(e.stats),
                this.checkForAbort_(e.requestId) || this.checkForIllegalMediaSwitch(t) || (t = t || {},
                function(e, t) {
                    if (!e && !t || !e && t || e && !t)
                        return !1;
                    if (e === t)
                        return !0;
                    const i = Object.keys(e).sort()
                      , s = Object.keys(t).sort();
                    if (i.length !== s.length)
                        return !1;
                    for (let n = 0; n < i.length; n++) {
                        const r = i[n];
                        if (r !== s[n])
                            return !1;
                        if (e[r] !== t[r])
                            return !1
                    }
                    return !0
                }(this.currentMediaInfo_, t) || (this.appendInitSegment_ = {
                    audio: !0,
                    video: !0
                },
                this.startingMediaInfo_ = t,
                this.currentMediaInfo_ = t,
                this.logger_("trackinfo update", t),
                this.trigger("trackinfo")),
                this.checkForAbort_(e.requestId) || (this.pendingSegment_.trackInfo = t,
                this.hasEnoughInfoToAppend_() && this.processCallQueue_()))
            }
            handleTimingInfo_(e, t, i, s) {
                if (this.earlyAbortWhenNeeded_(e.stats),
                this.checkForAbort_(e.requestId))
                    return;
                const n = this.pendingSegment_
                  , r = Fa(t);
                n[r] = n[r] || {},
                n[r][i] = s,
                this.logger_(`timinginfo: ${t} - ${i} - ${s}`),
                this.hasEnoughInfoToAppend_() && this.processCallQueue_()
            }
            handleCaptions_(e, t) {
                if (this.earlyAbortWhenNeeded_(e.stats),
                this.checkForAbort_(e.requestId))
                    return;
                if (0 === t.length)
                    return void this.logger_("SegmentLoader received no captions from a caption event");
                if (!this.pendingSegment_.hasAppendedData_)
                    return void this.metadataQueue_.caption.push(this.handleCaptions_.bind(this, e, t));
                const i = null === this.sourceUpdater_.videoTimestampOffset() ? this.sourceUpdater_.audioTimestampOffset() : this.sourceUpdater_.videoTimestampOffset()
                  , s = {};
                t.forEach((e=>{
                    s[e.stream] = s[e.stream] || {
                        startTime: 1 / 0,
                        captions: [],
                        endTime: 0
                    };
                    const t = s[e.stream];
                    t.startTime = Math.min(t.startTime, e.startTime + i),
                    t.endTime = Math.max(t.endTime, e.endTime + i),
                    t.captions.push(e)
                }
                )),
                Object.keys(s).forEach((e=>{
                    const {startTime: t, endTime: r, captions: a} = s[e]
                      , o = this.inbandTextTracks_;
                    this.logger_(`adding cues from ${t} -> ${r} for ${e}`),
                    function(e, t, i) {
                        if (!e[i]) {
                            t.trigger({
                                type: "usage",
                                name: "vhs-608"
                            });
                            let s = i;
                            /^cc708_/.test(i) && (s = "SERVICE" + i.split("_")[1]);
                            const n = t.textTracks().getTrackById(s);
                            if (n)
                                e[i] = n;
                            else {
                                let n = i
                                  , r = i
                                  , a = !1;
                                const o = (t.options_.vhs && t.options_.vhs.captionServices || {})[s];
                                o && (n = o.label,
                                r = o.language,
                                a = o.default),
                                e[i] = t.addRemoteTextTrack({
                                    kind: "captions",
                                    id: s,
                                    default: a,
                                    label: n,
                                    language: r
                                }, !1).track
                            }
                        }
                    }(o, this.vhs_.tech_, e),
                    Ra(t, r, o[e]),
                    function({inbandTextTracks: e, captionArray: t, timestampOffset: i}) {
                        if (!t)
                            return;
                        const s = n().WebKitDataCue || n().VTTCue;
                        t.forEach((t=>{
                            const n = t.stream;
                            t.content ? t.content.forEach((r=>{
                                const a = new s(t.startTime + i,t.endTime + i,r.text);
                                a.line = r.line,
                                a.align = "left",
                                a.position = r.position,
                                a.positionAlign = "line-left",
                                e[n].addCue(a)
                            }
                            )) : e[n].addCue(new s(t.startTime + i,t.endTime + i,t.text))
                        }
                        ))
                    }({
                        captionArray: a,
                        inbandTextTracks: o,
                        timestampOffset: i
                    })
                }
                )),
                this.transmuxer_ && this.transmuxer_.postMessage({
                    action: "clearParsedMp4Captions"
                })
            }
            handleId3_(e, t, i) {
                if (this.earlyAbortWhenNeeded_(e.stats),
                this.checkForAbort_(e.requestId))
                    return;
                this.pendingSegment_.hasAppendedData_ ? this.addMetadataToTextTrack(i, t, this.duration_()) : this.metadataQueue_.id3.push(this.handleId3_.bind(this, e, t, i))
            }
            processMetadataQueue_() {
                this.metadataQueue_.id3.forEach((e=>e())),
                this.metadataQueue_.caption.forEach((e=>e())),
                this.metadataQueue_.id3 = [],
                this.metadataQueue_.caption = []
            }
            processCallQueue_() {
                const e = this.callQueue_;
                this.callQueue_ = [],
                e.forEach((e=>e()))
            }
            processLoadQueue_() {
                const e = this.loadQueue_;
                this.loadQueue_ = [],
                e.forEach((e=>e()))
            }
            hasEnoughInfoToLoad_() {
                if ("audio" !== this.loaderType_)
                    return !0;
                const e = this.pendingSegment_;
                return !!e && (!this.getCurrentMediaInfo_() || !ja({
                    timelineChangeController: this.timelineChangeController_,
                    currentTimeline: this.currentTimeline_,
                    segmentTimeline: e.timeline,
                    loaderType: this.loaderType_,
                    audioDisabled: this.audioDisabled_
                }))
            }
            getCurrentMediaInfo_(e=this.pendingSegment_) {
                return e && e.trackInfo || this.currentMediaInfo_
            }
            getMediaInfo_(e=this.pendingSegment_) {
                return this.getCurrentMediaInfo_(e) || this.startingMediaInfo_
            }
            getPendingSegmentPlaylist() {
                return this.pendingSegment_ ? this.pendingSegment_.playlist : null
            }
            hasEnoughInfoToAppend_() {
                if (!this.sourceUpdater_.ready())
                    return !1;
                if (this.waitingOnRemove_ || this.quotaExceededErrorRetryTimeout_)
                    return !1;
                const e = this.pendingSegment_
                  , t = this.getCurrentMediaInfo_();
                if (!e || !t)
                    return !1;
                const {hasAudio: i, hasVideo: s, isMuxed: n} = t;
                return !(s && !e.videoTimingInfo) && (!(i && !this.audioDisabled_ && !n && !e.audioTimingInfo) && !ja({
                    timelineChangeController: this.timelineChangeController_,
                    currentTimeline: this.currentTimeline_,
                    segmentTimeline: e.timeline,
                    loaderType: this.loaderType_,
                    audioDisabled: this.audioDisabled_
                }))
            }
            handleData_(e, t) {
                if (this.earlyAbortWhenNeeded_(e.stats),
                this.checkForAbort_(e.requestId))
                    return;
                if (this.callQueue_.length || !this.hasEnoughInfoToAppend_())
                    return void this.callQueue_.push(this.handleData_.bind(this, e, t));
                const i = this.pendingSegment_;
                if (this.setTimeMapping_(i.timeline),
                this.updateMediaSecondsLoaded_(i.part || i.segment),
                "closed" !== this.mediaSource_.readyState) {
                    if (e.map && (e.map = this.initSegmentForMap(e.map, !0),
                    i.segment.map = e.map),
                    e.key && this.segmentKey(e.key, !0),
                    i.isFmp4 = e.isFmp4,
                    i.timingInfo = i.timingInfo || {},
                    i.isFmp4)
                        this.trigger("fmp4"),
                        i.timingInfo.start = i[Fa(t.type)].start;
                    else {
                        const e = this.getCurrentMediaInfo_()
                          , t = "main" === this.loaderType_ && e && e.hasVideo;
                        let s;
                        t && (s = i.videoTimingInfo.start),
                        i.timingInfo.start = this.trueSegmentStart_({
                            currentStart: i.timingInfo.start,
                            playlist: i.playlist,
                            mediaIndex: i.mediaIndex,
                            currentVideoTimestampOffset: this.sourceUpdater_.videoTimestampOffset(),
                            useVideoTimingInfo: t,
                            firstVideoFrameTimeForData: s,
                            videoTimingInfo: i.videoTimingInfo,
                            audioTimingInfo: i.audioTimingInfo
                        })
                    }
                    if (this.updateAppendInitSegmentStatus(i, t.type),
                    this.updateSourceBufferTimestampOffset_(i),
                    i.isSyncRequest) {
                        this.updateTimingInfoEnd_(i),
                        this.syncController_.saveSegmentTimingInfo({
                            segmentInfo: i,
                            shouldSaveTimelineMapping: "main" === this.loaderType_
                        });
                        const e = this.chooseNextRequest_();
                        if (e.mediaIndex !== i.mediaIndex || e.partIndex !== i.partIndex)
                            return void this.logger_("sync segment was incorrect, not appending");
                        this.logger_("sync segment was correct, appending")
                    }
                    i.hasAppendedData_ = !0,
                    this.processMetadataQueue_(),
                    this.appendData_(i, t)
                }
            }
            updateAppendInitSegmentStatus(e, t) {
                "main" !== this.loaderType_ || "number" !== typeof e.timestampOffset || e.changedTimestampOffset || (this.appendInitSegment_ = {
                    audio: !0,
                    video: !0
                }),
                this.playlistOfLastInitSegment_[t] !== e.playlist && (this.appendInitSegment_[t] = !0)
            }
            getInitSegmentAndUpdateState_({type: e, initSegment: t, map: i, playlist: s}) {
                if (i) {
                    const e = Dr(i);
                    if (this.activeInitSegmentId_ === e)
                        return null;
                    t = this.initSegmentForMap(i, !0).bytes,
                    this.activeInitSegmentId_ = e
                }
                return t && this.appendInitSegment_[e] ? (this.playlistOfLastInitSegment_[e] = s,
                this.appendInitSegment_[e] = !1,
                this.activeInitSegmentId_ = null,
                t) : null
            }
            handleQuotaExceededError_({segmentInfo: e, type: t, bytes: i}, s) {
                const r = this.sourceUpdater_.audioBuffered()
                  , a = this.sourceUpdater_.videoBuffered();
                r.length > 1 && this.logger_("On QUOTA_EXCEEDED_ERR, found gaps in the audio buffer: " + qn(r).join(", ")),
                a.length > 1 && this.logger_("On QUOTA_EXCEEDED_ERR, found gaps in the video buffer: " + qn(a).join(", "));
                const o = r.length ? r.start(0) : 0
                  , l = r.length ? r.end(r.length - 1) : 0
                  , h = a.length ? a.start(0) : 0
                  , d = a.length ? a.end(a.length - 1) : 0;
                if (l - o <= 1 && d - h <= 1)
                    return this.logger_(`On QUOTA_EXCEEDED_ERR, single segment too large to append to buffer, triggering an error. Appended byte length: ${i.byteLength}, audio buffer: ${qn(r).join(", ")}, video buffer: ${qn(a).join(", ")}, `),
                    this.error({
                        message: "Quota exceeded error with append of a single segment of content",
                        excludeUntil: 1 / 0
                    }),
                    void this.trigger("error");
                this.waitingOnRemove_ = !0,
                this.callQueue_.push(this.appendToSourceBuffer_.bind(this, {
                    segmentInfo: e,
                    type: t,
                    bytes: i
                }));
                const u = this.currentTime_() - 1;
                this.logger_(`On QUOTA_EXCEEDED_ERR, removing audio/video from 0 to ${u}`),
                this.remove(0, u, (()=>{
                    this.logger_("On QUOTA_EXCEEDED_ERR, retrying append in 1s"),
                    this.waitingOnRemove_ = !1,
                    this.quotaExceededErrorRetryTimeout_ = n().setTimeout((()=>{
                        this.logger_("On QUOTA_EXCEEDED_ERR, re-processing call queue"),
                        this.quotaExceededErrorRetryTimeout_ = null,
                        this.processCallQueue_()
                    }
                    ), 1e3)
                }
                ), !0)
            }
            handleAppendError_({segmentInfo: e, type: t, bytes: i}, s) {
                s && (22 !== s.code ? (this.logger_("Received non QUOTA_EXCEEDED_ERR on append", s),
                this.error(`${t} append of ${i.length}b failed for segment #${e.mediaIndex} in playlist ${e.playlist.id}`),
                this.trigger("appenderror")) : this.handleQuotaExceededError_({
                    segmentInfo: e,
                    type: t,
                    bytes: i
                }))
            }
            appendToSourceBuffer_({segmentInfo: e, type: t, initSegment: i, data: s, bytes: n}) {
                if (!n) {
                    const e = [s];
                    let t = s.byteLength;
                    i && (e.unshift(i),
                    t += i.byteLength),
                    n = (e=>{
                        let t, i = 0;
                        return e.bytes && (t = new Uint8Array(e.bytes),
                        e.segments.forEach((e=>{
                            t.set(e, i),
                            i += e.byteLength
                        }
                        ))),
                        t
                    }
                    )({
                        bytes: t,
                        segments: e
                    })
                }
                this.sourceUpdater_.appendBuffer({
                    segmentInfo: e,
                    type: t,
                    bytes: n
                }, this.handleAppendError_.bind(this, {
                    segmentInfo: e,
                    type: t,
                    bytes: n
                }))
            }
            handleSegmentTimingInfo_(e, t, i) {
                if (!this.pendingSegment_ || t !== this.pendingSegment_.requestId)
                    return;
                const s = this.pendingSegment_.segment
                  , n = `${e}TimingInfo`;
                s[n] || (s[n] = {}),
                s[n].transmuxerPrependedSeconds = i.prependedContentDuration || 0,
                s[n].transmuxedPresentationStart = i.start.presentation,
                s[n].transmuxedDecodeStart = i.start.decode,
                s[n].transmuxedPresentationEnd = i.end.presentation,
                s[n].transmuxedDecodeEnd = i.end.decode,
                s[n].baseMediaDecodeTime = i.baseMediaDecodeTime
            }
            appendData_(e, t) {
                const {type: i, data: s} = t;
                if (!s || !s.byteLength)
                    return;
                if ("audio" === i && this.audioDisabled_)
                    return;
                const n = this.getInitSegmentAndUpdateState_({
                    type: i,
                    initSegment: t.initSegment,
                    playlist: e.playlist,
                    map: e.isFmp4 ? e.segment.map : null
                });
                this.appendToSourceBuffer_({
                    segmentInfo: e,
                    type: i,
                    initSegment: n,
                    data: s
                })
            }
            loadSegment_(e) {
                this.state = "WAITING",
                this.pendingSegment_ = e,
                this.trimBackBuffer_(e),
                "number" === typeof e.timestampOffset && this.transmuxer_ && this.transmuxer_.postMessage({
                    action: "clearAllMp4Captions"
                }),
                this.hasEnoughInfoToLoad_() ? this.updateTransmuxerAndRequestSegment_(e) : this.loadQueue_.push((()=>{
                    const t = (0,
                    g.Z)({}, e, {
                        forceTimestampOffset: !0
                    });
                    (0,
                    g.Z)(e, this.generateSegmentInfo_(t)),
                    this.isPendingTimestampOffset_ = !1,
                    this.updateTransmuxerAndRequestSegment_(e)
                }
                ))
            }
            updateTransmuxerAndRequestSegment_(e) {
                this.shouldUpdateTransmuxerTimestampOffset_(e.timestampOffset) && (this.gopBuffer_.length = 0,
                e.gopsToAlignWith = [],
                this.timeMapping_ = 0,
                this.transmuxer_.postMessage({
                    action: "reset"
                }),
                this.transmuxer_.postMessage({
                    action: "setTimestampOffset",
                    timestampOffset: e.timestampOffset
                }));
                const t = this.createSimplifiedSegmentObj_(e)
                  , i = this.isEndOfStream_(e.mediaIndex, e.playlist, e.partIndex)
                  , s = null !== this.mediaIndex
                  , n = e.timeline !== this.currentTimeline_ && e.timeline > 0
                  , r = i || s && n;
                this.logger_(`Requesting ${Na(e)}`),
                t.map && !t.map.bytes && (this.logger_("going to request init segment."),
                this.appendInitSegment_ = {
                    video: !0,
                    audio: !0
                }),
                e.abortRequests = ya({
                    xhr: this.vhs_.xhr,
                    xhrOptions: this.xhrOptions_,
                    decryptionWorker: this.decrypter_,
                    segment: t,
                    abortFn: this.handleAbort_.bind(this, e),
                    progressFn: this.handleProgress_.bind(this),
                    trackInfoFn: this.handleTrackInfo_.bind(this),
                    timingInfoFn: this.handleTimingInfo_.bind(this),
                    videoSegmentTimingInfoFn: this.handleSegmentTimingInfo_.bind(this, "video", e.requestId),
                    audioSegmentTimingInfoFn: this.handleSegmentTimingInfo_.bind(this, "audio", e.requestId),
                    captionsFn: this.handleCaptions_.bind(this),
                    isEndOfTimeline: r,
                    endedTimelineFn: ()=>{
                        this.logger_("received endedtimeline callback")
                    }
                    ,
                    id3Fn: this.handleId3_.bind(this),
                    dataFn: this.handleData_.bind(this),
                    doneFn: this.segmentRequestFinished_.bind(this),
                    onTransmuxerLog: ({message: t, level: i, stream: s})=>{
                        this.logger_(`${Na(e)} logged from transmuxer stream ${s} as a ${i}: ${t}`)
                    }
                })
            }
            trimBackBuffer_(e) {
                const t = ((e,t,i)=>{
                    let s = t - zr.BACK_BUFFER_LENGTH;
                    e.length && (s = Math.max(s, e.start(0)));
                    const n = t - i;
                    return Math.min(n, s)
                }
                )(this.seekable_(), this.currentTime_(), this.playlist_.targetDuration || 10);
                t > 0 && this.remove(0, t)
            }
            createSimplifiedSegmentObj_(e) {
                const t = e.segment
                  , i = e.part
                  , s = {
                    resolvedUri: i ? i.resolvedUri : t.resolvedUri,
                    byterange: i ? i.byterange : t.byterange,
                    requestId: e.requestId,
                    transmuxer: e.transmuxer,
                    audioAppendStart: e.audioAppendStart,
                    gopsToAlignWith: e.gopsToAlignWith,
                    part: e.part
                }
                  , n = e.playlist.segments[e.mediaIndex - 1];
                if (n && n.timeline === t.timeline && (n.videoTimingInfo ? s.baseStartTime = n.videoTimingInfo.transmuxedDecodeEnd : n.audioTimingInfo && (s.baseStartTime = n.audioTimingInfo.transmuxedDecodeEnd)),
                t.key) {
                    const i = t.key.iv || new Uint32Array([0, 0, 0, e.mediaIndex + e.playlist.mediaSequence]);
                    s.key = this.segmentKey(t.key),
                    s.key.iv = i
                }
                return t.map && (s.map = this.initSegmentForMap(t.map)),
                s
            }
            saveTransferStats_(e) {
                this.mediaRequests += 1,
                e && (this.mediaBytesTransferred += e.bytesReceived,
                this.mediaTransferDuration += e.roundTripTime)
            }
            saveBandwidthRelatedStats_(e, t) {
                this.pendingSegment_.byteLength = t.bytesReceived,
                e < Ba ? this.logger_(`Ignoring segment's bandwidth because its duration of ${e} is less than the min to record 0.016666666666666666`) : (this.bandwidth = t.bandwidth,
                this.roundTrip = t.roundTripTime)
            }
            handleTimeout_() {
                this.mediaRequestsTimedout += 1,
                this.bandwidth = 1,
                this.roundTrip = NaN,
                this.trigger("bandwidthupdate"),
                this.trigger("timeout")
            }
            segmentRequestFinished_(e, t, i) {
                if (this.callQueue_.length)
                    return void this.callQueue_.push(this.segmentRequestFinished_.bind(this, e, t, i));
                if (this.saveTransferStats_(t.stats),
                !this.pendingSegment_)
                    return;
                if (t.requestId !== this.pendingSegment_.requestId)
                    return;
                if (e) {
                    if (this.pendingSegment_ = null,
                    this.state = "READY",
                    e.code === oa)
                        return;
                    return this.pause(),
                    e.code === aa ? void this.handleTimeout_() : (this.mediaRequestsErrored += 1,
                    this.error(e),
                    void this.trigger("error"))
                }
                const s = this.pendingSegment_;
                this.saveBandwidthRelatedStats_(s.duration, t.stats),
                s.endOfAllRequests = t.endOfAllRequests,
                i.gopInfo && (this.gopBuffer_ = ((e,t,i)=>{
                    if (!t.length)
                        return e;
                    if (i)
                        return t.slice();
                    const s = t[0].pts;
                    let n = 0;
                    for (; n < e.length && !(e[n].pts >= s); n++)
                        ;
                    return e.slice(0, n).concat(t)
                }
                )(this.gopBuffer_, i.gopInfo, this.safeAppend_)),
                this.state = "APPENDING",
                this.trigger("appending"),
                this.waitForAppendsToComplete_(s)
            }
            setTimeMapping_(e) {
                const t = this.syncController_.mappingForTimeline(e);
                null !== t && (this.timeMapping_ = t)
            }
            updateMediaSecondsLoaded_(e) {
                "number" === typeof e.start && "number" === typeof e.end ? this.mediaSecondsLoaded += e.end - e.start : this.mediaSecondsLoaded += e.duration
            }
            shouldUpdateTransmuxerTimestampOffset_(e) {
                return null !== e && ("main" === this.loaderType_ && e !== this.sourceUpdater_.videoTimestampOffset() || !this.audioDisabled_ && e !== this.sourceUpdater_.audioTimestampOffset())
            }
            trueSegmentStart_({currentStart: e, playlist: t, mediaIndex: i, firstVideoFrameTimeForData: s, currentVideoTimestampOffset: n, useVideoTimingInfo: r, videoTimingInfo: a, audioTimingInfo: o}) {
                if ("undefined" !== typeof e)
                    return e;
                if (!r)
                    return o.start;
                const l = t.segments[i - 1];
                return 0 !== i && l && "undefined" !== typeof l.start && l.end === s + n ? a.start : s
            }
            waitForAppendsToComplete_(e) {
                const t = this.getCurrentMediaInfo_(e);
                if (!t)
                    return this.error({
                        message: "No starting media returned, likely due to an unsupported media format.",
                        playlistExclusionDuration: 1 / 0
                    }),
                    void this.trigger("error");
                const {hasAudio: i, hasVideo: s, isMuxed: n} = t
                  , r = "main" === this.loaderType_ && s
                  , a = !this.audioDisabled_ && i && !n;
                if (e.waitingOnAppends = 0,
                !e.hasAppendedData_)
                    return e.timingInfo || "number" !== typeof e.timestampOffset || (this.isPendingTimestampOffset_ = !0),
                    e.timingInfo = {
                        start: 0
                    },
                    e.waitingOnAppends++,
                    this.isPendingTimestampOffset_ || (this.updateSourceBufferTimestampOffset_(e),
                    this.processMetadataQueue_()),
                    void this.checkAppendsDone_(e);
                r && e.waitingOnAppends++,
                a && e.waitingOnAppends++,
                r && this.sourceUpdater_.videoQueueCallback(this.checkAppendsDone_.bind(this, e)),
                a && this.sourceUpdater_.audioQueueCallback(this.checkAppendsDone_.bind(this, e))
            }
            checkAppendsDone_(e) {
                this.checkForAbort_(e.requestId) || (e.waitingOnAppends--,
                0 === e.waitingOnAppends && this.handleAppendsDone_())
            }
            checkForIllegalMediaSwitch(e) {
                const t = ((e,t,i)=>"main" === e && t && i ? i.hasAudio || i.hasVideo ? t.hasVideo && !i.hasVideo ? "Only audio found in segment when we expected video. We can't switch to audio only from a stream that had video. To get rid of this message, please add codec information to the manifest." : !t.hasVideo && i.hasVideo ? "Video found in segment when we expected only audio. We can't switch to a stream with video from an audio only stream. To get rid of this message, please add codec information to the manifest." : null : "Neither audio nor video found in segment." : null)(this.loaderType_, this.getCurrentMediaInfo_(), e);
                return !!t && (this.error({
                    message: t,
                    playlistExclusionDuration: 1 / 0
                }),
                this.trigger("error"),
                !0)
            }
            updateSourceBufferTimestampOffset_(e) {
                if (null === e.timestampOffset || "number" !== typeof e.timingInfo.start || e.changedTimestampOffset || "main" !== this.loaderType_)
                    return;
                let t = !1;
                e.timestampOffset -= this.getSegmentStartTimeForTimestampOffsetCalculation_({
                    videoTimingInfo: e.segment.videoTimingInfo,
                    audioTimingInfo: e.segment.audioTimingInfo,
                    timingInfo: e.timingInfo
                }),
                e.changedTimestampOffset = !0,
                e.timestampOffset !== this.sourceUpdater_.videoTimestampOffset() && (this.sourceUpdater_.videoTimestampOffset(e.timestampOffset),
                t = !0),
                e.timestampOffset !== this.sourceUpdater_.audioTimestampOffset() && (this.sourceUpdater_.audioTimestampOffset(e.timestampOffset),
                t = !0),
                t && this.trigger("timestampoffset")
            }
            getSegmentStartTimeForTimestampOffsetCalculation_({videoTimingInfo: e, audioTimingInfo: t, timingInfo: i}) {
                return this.useDtsForTimestampOffset_ ? e && "number" === typeof e.transmuxedDecodeStart ? e.transmuxedDecodeStart : t && "number" === typeof t.transmuxedDecodeStart ? t.transmuxedDecodeStart : i.start : i.start
            }
            updateTimingInfoEnd_(e) {
                e.timingInfo = e.timingInfo || {};
                const t = this.getMediaInfo_()
                  , i = "main" === this.loaderType_ && t && t.hasVideo && e.videoTimingInfo ? e.videoTimingInfo : e.audioTimingInfo;
                i && (e.timingInfo.end = "number" === typeof i.end ? i.end : i.start + e.duration)
            }
            handleAppendsDone_() {
                if (this.pendingSegment_ && this.trigger("appendsdone"),
                !this.pendingSegment_)
                    return this.state = "READY",
                    void (this.paused() || this.monitorBuffer_());
                const e = this.pendingSegment_;
                this.updateTimingInfoEnd_(e),
                this.shouldSaveSegmentTimingInfo_ && this.syncController_.saveSegmentTimingInfo({
                    segmentInfo: e,
                    shouldSaveTimelineMapping: "main" === this.loaderType_
                });
                const t = qa(e, this.sourceType_);
                if (t && ("warn" === t.severity ? wn.log.warn(t.message) : this.logger_(t.message)),
                this.recordThroughput_(e),
                this.pendingSegment_ = null,
                this.state = "READY",
                e.isSyncRequest && (this.trigger("syncinfoupdate"),
                !e.hasAppendedData_))
                    return void this.logger_(`Throwing away un-appended sync request ${Na(e)}`);
                this.logger_(`Appended ${Na(e)}`),
                this.addSegmentMetadataCue_(e),
                this.fetchAtBuffer_ = !0,
                this.currentTimeline_ !== e.timeline && (this.timelineChangeController_.lastTimelineChange({
                    type: this.loaderType_,
                    from: this.currentTimeline_,
                    to: e.timeline
                }),
                "main" !== this.loaderType_ || this.audioDisabled_ || this.timelineChangeController_.lastTimelineChange({
                    type: "audio",
                    from: this.currentTimeline_,
                    to: e.timeline
                })),
                this.currentTimeline_ = e.timeline,
                this.trigger("syncinfoupdate");
                const i = e.segment
                  , s = e.part
                  , n = i.end && this.currentTime_() - i.end > 3 * e.playlist.targetDuration
                  , r = s && s.end && this.currentTime_() - s.end > 3 * e.playlist.partTargetDuration;
                if (n || r)
                    return this.logger_(`bad ${n ? "segment" : "part"} ${Na(e)}`),
                    void this.resetEverything();
                null !== this.mediaIndex && this.trigger("bandwidthupdate"),
                this.trigger("progress"),
                this.mediaIndex = e.mediaIndex,
                this.partIndex = e.partIndex,
                this.isEndOfStream_(e.mediaIndex, e.playlist, e.partIndex) && this.endOfStream(),
                this.trigger("appended"),
                e.hasAppendedData_ && this.mediaAppends++,
                this.paused() || this.monitorBuffer_()
            }
            recordThroughput_(e) {
                if (e.duration < Ba)
                    return void this.logger_(`Ignoring segment's throughput because its duration of ${e.duration} is less than the min to record 0.016666666666666666`);
                const t = this.throughput.rate
                  , i = Date.now() - e.endOfAllRequests + 1
                  , s = Math.floor(e.byteLength / i * 8 * 1e3);
                this.throughput.rate += (s - t) / ++this.throughput.count
            }
            addSegmentMetadataCue_(e) {
                if (!this.segmentMetadataTrack_)
                    return;
                const t = e.segment
                  , i = t.start
                  , s = t.end;
                if (!Ua(i) || !Ua(s))
                    return;
                Ra(i, s, this.segmentMetadataTrack_);
                const r = n().WebKitDataCue || n().VTTCue
                  , a = {
                    custom: t.custom,
                    dateTimeObject: t.dateTimeObject,
                    dateTimeString: t.dateTimeString,
                    programDateTime: t.programDateTime,
                    bandwidth: e.playlist.attributes.BANDWIDTH,
                    resolution: e.playlist.attributes.RESOLUTION,
                    codecs: e.playlist.attributes.CODECS,
                    byteLength: e.byteLength,
                    uri: e.uri,
                    timeline: e.timeline,
                    playlist: e.playlist.id,
                    start: i,
                    end: s
                }
                  , o = new r(i,s,JSON.stringify(a));
                o.value = a,
                this.segmentMetadataTrack_.addCue(o)
            }
        }
        function Va() {}
        const za = function(e) {
            return "string" !== typeof e ? e : e.replace(/./, (e=>e.toUpperCase()))
        }
          , Wa = ["video", "audio"]
          , Ga = (e,t)=>{
            const i = t[`${e}Buffer`];
            return i && i.updating || t.queuePending[e]
        }
          , Ka = (e,t)=>{
            if (0 === t.queue.length)
                return;
            let i = 0
              , s = t.queue[i];
            if ("mediaSource" !== s.type) {
                if ("mediaSource" !== e && t.ready() && "closed" !== t.mediaSource.readyState && !Ga(e, t)) {
                    if (s.type !== e) {
                        if (i = ((e,t)=>{
                            for (let i = 0; i < t.length; i++) {
                                const s = t[i];
                                if ("mediaSource" === s.type)
                                    return null;
                                if (s.type === e)
                                    return i
                            }
                            return null
                        }
                        )(e, t.queue),
                        null === i)
                            return;
                        s = t.queue[i]
                    }
                    return t.queue.splice(i, 1),
                    t.queuePending[e] = s,
                    s.action(e, t),
                    s.doneFn ? void 0 : (t.queuePending[e] = null,
                    void Ka(e, t))
                }
            } else
                t.updating() || "closed" === t.mediaSource.readyState || (t.queue.shift(),
                s.action(t),
                s.doneFn && s.doneFn(),
                Ka("audio", t),
                Ka("video", t))
        }
          , Qa = (e,t)=>{
            const i = t[`${e}Buffer`]
              , s = za(e);
            i && (i.removeEventListener("updateend", t[`on${s}UpdateEnd_`]),
            i.removeEventListener("error", t[`on${s}Error_`]),
            t.codecs[e] = null,
            t[`${e}Buffer`] = null)
        }
          , Xa = (e,t)=>e && t && -1 !== Array.prototype.indexOf.call(e.sourceBuffers, t)
          , Ya = (e,t,i)=>(s,n)=>{
            const r = n[`${s}Buffer`];
            if (Xa(n.mediaSource, r)) {
                n.logger_(`Appending segment ${t.mediaIndex}'s ${e.length} bytes to ${s}Buffer`);
                try {
                    r.appendBuffer(e)
                } catch (a) {
                    n.logger_(`Error with code ${a.code} ` + (22 === a.code ? "(QUOTA_EXCEEDED_ERR) " : "") + `when appending segment ${t.mediaIndex} to ${s}Buffer`),
                    n.queuePending[s] = null,
                    i(a)
                }
            }
        }
          , Ja = (e,t)=>(i,s)=>{
            const n = s[`${i}Buffer`];
            if (Xa(s.mediaSource, n)) {
                s.logger_(`Removing ${e} to ${t} from ${i}Buffer`);
                try {
                    n.remove(e, t)
                } catch (r) {
                    s.logger_(`Remove ${e} to ${t} from ${i}Buffer failed`)
                }
            }
        }
          , Za = e=>(t,i)=>{
            const s = i[`${t}Buffer`];
            Xa(i.mediaSource, s) && (i.logger_(`Setting ${t}timestampOffset to ${e}`),
            s.timestampOffset = e)
        }
          , eo = e=>(t,i)=>{
            e()
        }
          , to = e=>t=>{
            if ("open" === t.mediaSource.readyState) {
                t.logger_(`Calling mediaSource endOfStream(${e || ""})`);
                try {
                    t.mediaSource.endOfStream(e)
                } catch (i) {
                    wn.log.warn("Failed to call media source endOfStream", i)
                }
            }
        }
          , io = e=>t=>{
            t.logger_(`Setting mediaSource duration to ${e}`);
            try {
                t.mediaSource.duration = e
            } catch (i) {
                wn.log.warn("Failed to set media source duration", i)
            }
        }
          , so = ()=>(e,t)=>{
            if ("open" !== t.mediaSource.readyState)
                return;
            const i = t[`${e}Buffer`];
            if (Xa(t.mediaSource, i)) {
                t.logger_(`calling abort on ${e}Buffer`);
                try {
                    i.abort()
                } catch (s) {
                    wn.log.warn(`Failed to abort on ${e}Buffer`, s)
                }
            }
        }
          , no = (e,t)=>i=>{
            const s = za(e)
              , n = (0,
            y._5)(t);
            i.logger_(`Adding ${e}Buffer with codec ${t} to mediaSource`);
            const r = i.mediaSource.addSourceBuffer(n);
            r.addEventListener("updateend", i[`on${s}UpdateEnd_`]),
            r.addEventListener("error", i[`on${s}Error_`]),
            i.codecs[e] = t,
            i[`${e}Buffer`] = r
        }
          , ro = e=>t=>{
            const i = t[`${e}Buffer`];
            if (Qa(e, t),
            Xa(t.mediaSource, i)) {
                t.logger_(`Removing ${e}Buffer with codec ${t.codecs[e]} from mediaSource`);
                try {
                    t.mediaSource.removeSourceBuffer(i)
                } catch (s) {
                    wn.log.warn(`Failed to removeSourceBuffer ${e}Buffer`, s)
                }
            }
        }
          , ao = e=>(t,i)=>{
            const s = i[`${t}Buffer`]
              , n = (0,
            y._5)(e);
            if (Xa(i.mediaSource, s) && i.codecs[t] !== e) {
                i.logger_(`changing ${t}Buffer codec from ${i.codecs[t]} to ${e}`);
                try {
                    s.changeType(n),
                    i.codecs[t] = e
                } catch (r) {
                    wn.log.warn(`Failed to changeType on ${t}Buffer`, r)
                }
            }
        }
          , oo = ({type: e, sourceUpdater: t, action: i, doneFn: s, name: n})=>{
            t.queue.push({
                type: e,
                action: i,
                doneFn: s,
                name: n
            }),
            Ka(e, t)
        }
          , lo = (e,t)=>i=>{
            if (t.queuePending[e]) {
                const i = t.queuePending[e].doneFn;
                t.queuePending[e] = null,
                i && i(t[`${e}Error_`])
            }
            Ka(e, t)
        }
        ;
        class ho extends wn.EventTarget {
            constructor(e) {
                super(),
                this.mediaSource = e,
                this.sourceopenListener_ = ()=>Ka("mediaSource", this),
                this.mediaSource.addEventListener("sourceopen", this.sourceopenListener_),
                this.logger_ = On("SourceUpdater"),
                this.audioTimestampOffset_ = 0,
                this.videoTimestampOffset_ = 0,
                this.queue = [],
                this.queuePending = {
                    audio: null,
                    video: null
                },
                this.delayedAudioAppendQueue_ = [],
                this.videoAppendQueued_ = !1,
                this.codecs = {},
                this.onVideoUpdateEnd_ = lo("video", this),
                this.onAudioUpdateEnd_ = lo("audio", this),
                this.onVideoError_ = e=>{
                    this.videoError_ = e
                }
                ,
                this.onAudioError_ = e=>{
                    this.audioError_ = e
                }
                ,
                this.createdSourceBuffers_ = !1,
                this.initializedEme_ = !1,
                this.triggeredReady_ = !1
            }
            initializedEme() {
                this.initializedEme_ = !0,
                this.triggerReady()
            }
            hasCreatedSourceBuffers() {
                return this.createdSourceBuffers_
            }
            hasInitializedAnyEme() {
                return this.initializedEme_
            }
            ready() {
                return this.hasCreatedSourceBuffers() && this.hasInitializedAnyEme()
            }
            createSourceBuffers(e) {
                this.hasCreatedSourceBuffers() || (this.addOrChangeSourceBuffers(e),
                this.createdSourceBuffers_ = !0,
                this.trigger("createdsourcebuffers"),
                this.triggerReady())
            }
            triggerReady() {
                this.ready() && !this.triggeredReady_ && (this.triggeredReady_ = !0,
                this.trigger("ready"))
            }
            addSourceBuffer(e, t) {
                oo({
                    type: "mediaSource",
                    sourceUpdater: this,
                    action: no(e, t),
                    name: "addSourceBuffer"
                })
            }
            abort(e) {
                oo({
                    type: e,
                    sourceUpdater: this,
                    action: so(e),
                    name: "abort"
                })
            }
            removeSourceBuffer(e) {
                this.canRemoveSourceBuffer() ? oo({
                    type: "mediaSource",
                    sourceUpdater: this,
                    action: ro(e),
                    name: "removeSourceBuffer"
                }) : wn.log.error("removeSourceBuffer is not supported!")
            }
            canRemoveSourceBuffer() {
                return !wn.browser.IS_FIREFOX && n().MediaSource && n().MediaSource.prototype && "function" === typeof n().MediaSource.prototype.removeSourceBuffer
            }
            static canChangeType() {
                return n().SourceBuffer && n().SourceBuffer.prototype && "function" === typeof n().SourceBuffer.prototype.changeType
            }
            canChangeType() {
                return this.constructor.canChangeType()
            }
            changeType(e, t) {
                this.canChangeType() ? oo({
                    type: e,
                    sourceUpdater: this,
                    action: ao(t),
                    name: "changeType"
                }) : wn.log.error("changeType is not supported!")
            }
            addOrChangeSourceBuffers(e) {
                if (!e || "object" !== typeof e || 0 === Object.keys(e).length)
                    throw new Error("Cannot addOrChangeSourceBuffers to undefined codecs");
                Object.keys(e).forEach((t=>{
                    const i = e[t];
                    if (!this.hasCreatedSourceBuffers())
                        return this.addSourceBuffer(t, i);
                    this.canChangeType() && this.changeType(t, i)
                }
                ))
            }
            appendBuffer(e, t) {
                const {segmentInfo: i, type: s, bytes: n} = e;
                if (this.processedAppend_ = !0,
                "audio" === s && this.videoBuffer && !this.videoAppendQueued_)
                    return this.delayedAudioAppendQueue_.push([e, t]),
                    void this.logger_(`delayed audio append of ${n.length} until video append`);
                if (oo({
                    type: s,
                    sourceUpdater: this,
                    action: Ya(n, i || {
                        mediaIndex: -1
                    }, t),
                    doneFn: t,
                    name: "appendBuffer"
                }),
                "video" === s) {
                    if (this.videoAppendQueued_ = !0,
                    !this.delayedAudioAppendQueue_.length)
                        return;
                    const e = this.delayedAudioAppendQueue_.slice();
                    this.logger_(`queuing delayed audio ${e.length} appendBuffers`),
                    this.delayedAudioAppendQueue_.length = 0,
                    e.forEach((e=>{
                        this.appendBuffer.apply(this, e)
                    }
                    ))
                }
            }
            audioBuffered() {
                return Xa(this.mediaSource, this.audioBuffer) && this.audioBuffer.buffered ? this.audioBuffer.buffered : Rn()
            }
            videoBuffered() {
                return Xa(this.mediaSource, this.videoBuffer) && this.videoBuffer.buffered ? this.videoBuffer.buffered : Rn()
            }
            buffered() {
                const e = Xa(this.mediaSource, this.videoBuffer) ? this.videoBuffer : null
                  , t = Xa(this.mediaSource, this.audioBuffer) ? this.audioBuffer : null;
                return t && !e ? this.audioBuffered() : e && !t ? this.videoBuffered() : function(e, t) {
                    let i = null
                      , s = null
                      , n = 0;
                    const r = []
                      , a = [];
                    if (!e || !e.length || !t || !t.length)
                        return Rn();
                    let o = e.length;
                    for (; o--; )
                        r.push({
                            time: e.start(o),
                            type: "start"
                        }),
                        r.push({
                            time: e.end(o),
                            type: "end"
                        });
                    for (o = t.length; o--; )
                        r.push({
                            time: t.start(o),
                            type: "start"
                        }),
                        r.push({
                            time: t.end(o),
                            type: "end"
                        });
                    for (r.sort((function(e, t) {
                        return e.time - t.time
                    }
                    )),
                    o = 0; o < r.length; o++)
                        "start" === r[o].type ? (n++,
                        2 === n && (i = r[o].time)) : "end" === r[o].type && (n--,
                        1 === n && (s = r[o].time)),
                        null !== i && null !== s && (a.push([i, s]),
                        i = null,
                        s = null);
                    return Rn(a)
                }(this.audioBuffered(), this.videoBuffered())
            }
            setDuration(e, t=Va) {
                oo({
                    type: "mediaSource",
                    sourceUpdater: this,
                    action: io(e),
                    name: "duration",
                    doneFn: t
                })
            }
            endOfStream(e=null, t=Va) {
                "string" !== typeof e && (e = void 0),
                oo({
                    type: "mediaSource",
                    sourceUpdater: this,
                    action: to(e),
                    name: "endOfStream",
                    doneFn: t
                })
            }
            removeAudio(e, t, i=Va) {
                this.audioBuffered().length && 0 !== this.audioBuffered().end(0) ? oo({
                    type: "audio",
                    sourceUpdater: this,
                    action: Ja(e, t),
                    doneFn: i,
                    name: "remove"
                }) : i()
            }
            removeVideo(e, t, i=Va) {
                this.videoBuffered().length && 0 !== this.videoBuffered().end(0) ? oo({
                    type: "video",
                    sourceUpdater: this,
                    action: Ja(e, t),
                    doneFn: i,
                    name: "remove"
                }) : i()
            }
            updating() {
                return !(!Ga("audio", this) && !Ga("video", this))
            }
            audioTimestampOffset(e) {
                return "undefined" !== typeof e && this.audioBuffer && this.audioTimestampOffset_ !== e && (oo({
                    type: "audio",
                    sourceUpdater: this,
                    action: Za(e),
                    name: "timestampOffset"
                }),
                this.audioTimestampOffset_ = e),
                this.audioTimestampOffset_
            }
            videoTimestampOffset(e) {
                return "undefined" !== typeof e && this.videoBuffer && this.videoTimestampOffset !== e && (oo({
                    type: "video",
                    sourceUpdater: this,
                    action: Za(e),
                    name: "timestampOffset"
                }),
                this.videoTimestampOffset_ = e),
                this.videoTimestampOffset_
            }
            audioQueueCallback(e) {
                this.audioBuffer && oo({
                    type: "audio",
                    sourceUpdater: this,
                    action: eo(e),
                    name: "callback"
                })
            }
            videoQueueCallback(e) {
                this.videoBuffer && oo({
                    type: "video",
                    sourceUpdater: this,
                    action: eo(e),
                    name: "callback"
                })
            }
            dispose() {
                this.trigger("dispose"),
                Wa.forEach((e=>{
                    this.abort(e),
                    this.canRemoveSourceBuffer() ? this.removeSourceBuffer(e) : this[`${e}QueueCallback`]((()=>Qa(e, this)))
                }
                )),
                this.videoAppendQueued_ = !1,
                this.delayedAudioAppendQueue_.length = 0,
                this.sourceopenListener_ && this.mediaSource.removeEventListener("sourceopen", this.sourceopenListener_),
                this.off()
            }
        }
        const uo = e=>decodeURIComponent(escape(String.fromCharCode.apply(null, e)))
          , co = new Uint8Array("\n\n".split("").map((e=>e.charCodeAt(0))));
        class po extends Error {
            constructor() {
                super("Trying to parse received VTT cues, but there is no WebVTT. Make sure vtt.js is loaded.")
            }
        }
        class mo extends Ha {
            constructor(e, t={}) {
                super(e, t),
                this.mediaSource_ = null,
                this.subtitlesTrack_ = null,
                this.loaderType_ = "subtitle",
                this.featuresNativeTextTracks_ = e.featuresNativeTextTracks,
                this.loadVttJs = e.loadVttJs,
                this.shouldSaveSegmentTimingInfo_ = !1
            }
            createTransmuxer_() {
                return null
            }
            buffered_() {
                if (!this.subtitlesTrack_ || !this.subtitlesTrack_.cues || !this.subtitlesTrack_.cues.length)
                    return Rn();
                const e = this.subtitlesTrack_.cues;
                return Rn([[e[0].startTime, e[e.length - 1].startTime]])
            }
            initSegmentForMap(e, t=!1) {
                if (!e)
                    return null;
                const i = Dr(e);
                let s = this.initSegments_[i];
                if (t && !s && e.bytes) {
                    const t = co.byteLength + e.bytes.byteLength
                      , n = new Uint8Array(t);
                    n.set(e.bytes),
                    n.set(co, e.bytes.byteLength),
                    this.initSegments_[i] = s = {
                        resolvedUri: e.resolvedUri,
                        byterange: e.byterange,
                        bytes: n
                    }
                }
                return s || e
            }
            couldBeginLoading_() {
                return this.playlist_ && this.subtitlesTrack_ && !this.paused()
            }
            init_() {
                return this.state = "READY",
                this.resetEverything(),
                this.monitorBuffer_()
            }
            track(e) {
                return "undefined" === typeof e || (this.subtitlesTrack_ = e,
                "INIT" === this.state && this.couldBeginLoading_() && this.init_()),
                this.subtitlesTrack_
            }
            remove(e, t) {
                Ra(e, t, this.subtitlesTrack_)
            }
            fillBuffer_() {
                const e = this.chooseNextRequest_();
                if (e) {
                    if (null === this.syncController_.timestampOffsetForTimeline(e.timeline)) {
                        const e = ()=>{
                            this.state = "READY",
                            this.paused() || this.monitorBuffer_()
                        }
                        ;
                        return this.syncController_.one("timestampoffset", e),
                        void (this.state = "WAITING_ON_TIMELINE")
                    }
                    this.loadSegment_(e)
                }
            }
            timestampOffsetForSegment_() {
                return null
            }
            chooseNextRequest_() {
                return this.skipEmptySegments_(super.chooseNextRequest_())
            }
            skipEmptySegments_(e) {
                for (; e && e.segment.empty; ) {
                    if (e.mediaIndex + 1 >= e.playlist.segments.length) {
                        e = null;
                        break
                    }
                    e = this.generateSegmentInfo_({
                        playlist: e.playlist,
                        mediaIndex: e.mediaIndex + 1,
                        startOfSegment: e.startOfSegment + e.duration,
                        isSyncRequest: e.isSyncRequest
                    })
                }
                return e
            }
            stopForError(e) {
                this.error(e),
                this.state = "READY",
                this.pause(),
                this.trigger("error")
            }
            segmentRequestFinished_(e, t, i) {
                if (!this.subtitlesTrack_)
                    return void (this.state = "READY");
                if (this.saveTransferStats_(t.stats),
                !this.pendingSegment_)
                    return this.state = "READY",
                    void (this.mediaRequestsAborted += 1);
                if (e)
                    return e.code === aa && this.handleTimeout_(),
                    e.code === oa ? this.mediaRequestsAborted += 1 : this.mediaRequestsErrored += 1,
                    void this.stopForError(e);
                const s = this.pendingSegment_;
                this.saveBandwidthRelatedStats_(s.duration, t.stats),
                t.key && this.segmentKey(t.key, !0),
                this.state = "APPENDING",
                this.trigger("appending");
                const r = s.segment;
                if (r.map && (r.map.bytes = t.map.bytes),
                s.bytes = t.bytes,
                "function" !== typeof n().WebVTT && "function" === typeof this.loadVttJs)
                    return this.state = "WAITING_ON_VTTJS",
                    void this.loadVttJs().then((()=>this.segmentRequestFinished_(e, t, i)), (()=>this.stopForError({
                        message: "Error loading vtt.js"
                    })));
                r.requested = !0;
                try {
                    this.parseVTTCues_(s)
                } catch (a) {
                    return void this.stopForError({
                        message: a.message
                    })
                }
                if (this.updateTimeMapping_(s, this.syncController_.timelines[s.timeline], this.playlist_),
                s.cues.length ? s.timingInfo = {
                    start: s.cues[0].startTime,
                    end: s.cues[s.cues.length - 1].endTime
                } : s.timingInfo = {
                    start: s.startOfSegment,
                    end: s.startOfSegment + s.duration
                },
                s.isSyncRequest)
                    return this.trigger("syncinfoupdate"),
                    this.pendingSegment_ = null,
                    void (this.state = "READY");
                s.byteLength = s.bytes.byteLength,
                this.mediaSecondsLoaded += r.duration,
                s.cues.forEach((e=>{
                    this.subtitlesTrack_.addCue(this.featuresNativeTextTracks_ ? new (n().VTTCue)(e.startTime,e.endTime,e.text) : e)
                }
                )),
                function(e) {
                    const t = e.cues;
                    if (!t)
                        return;
                    const i = {};
                    for (let s = t.length - 1; s >= 0; s--) {
                        const n = t[s]
                          , r = `${n.startTime}-${n.endTime}-${n.text}`;
                        i[r] ? e.removeCue(n) : i[r] = n
                    }
                }(this.subtitlesTrack_),
                this.handleAppendsDone_()
            }
            handleData_() {}
            updateTimingInfoEnd_() {}
            parseVTTCues_(e) {
                let t, i = !1;
                if ("function" !== typeof n().WebVTT)
                    throw new po;
                "function" === typeof n().TextDecoder ? t = new (n().TextDecoder)("utf8") : (t = n().WebVTT.StringDecoder(),
                i = !0);
                const s = new (n().WebVTT.Parser)(n(),n().vttjs,t);
                if (e.cues = [],
                e.timestampmap = {
                    MPEGTS: 0,
                    LOCAL: 0
                },
                s.oncue = e.cues.push.bind(e.cues),
                s.ontimestampmap = t=>{
                    e.timestampmap = t
                }
                ,
                s.onparsingerror = e=>{
                    wn.log.warn("Error encountered when parsing cues: " + e.message)
                }
                ,
                e.segment.map) {
                    let t = e.segment.map.bytes;
                    i && (t = uo(t)),
                    s.parse(t)
                }
                let r = e.bytes;
                i && (r = uo(r)),
                s.parse(r),
                s.flush()
            }
            updateTimeMapping_(e, t, i) {
                const s = e.segment;
                if (!t)
                    return;
                if (!e.cues.length)
                    return void (s.empty = !0);
                const {MPEGTS: n, LOCAL: r} = e.timestampmap
                  , a = n / w.ONE_SECOND_IN_TS - r + t.mapping;
                if (e.cues.forEach((e=>{
                    const i = e.endTime - e.startTime
                      , s = 0 === n ? e.startTime + a : this.handleRollover_(e.startTime + a, t.time);
                    e.startTime = Math.max(s, 0),
                    e.endTime = Math.max(s + i, 0)
                }
                )),
                !i.syncInfo) {
                    const t = e.cues[0].startTime
                      , n = e.cues[e.cues.length - 1].startTime;
                    i.syncInfo = {
                        mediaSequence: i.mediaSequence + e.mediaIndex,
                        time: Math.min(t, n - s.duration)
                    }
                }
            }
            handleRollover_(e, t) {
                if (null === t)
                    return e;
                let i = e * w.ONE_SECOND_IN_TS;
                const s = t * w.ONE_SECOND_IN_TS;
                let n;
                for (n = s < i ? -8589934592 : 8589934592; Math.abs(i - s) > 4294967296; )
                    i += n;
                return i / w.ONE_SECOND_IN_TS
            }
        }
        const go = function(e, t) {
            const i = e.cues;
            for (let s = 0; s < i.length; s++) {
                const e = i[s];
                if (t >= e.adStartTime && t <= e.adEndTime)
                    return e
            }
            return null
        }
          , fo = [{
            name: "VOD",
            run: (e,t,i,s,n)=>{
                if (i !== 1 / 0) {
                    return {
                        time: 0,
                        segmentIndex: 0,
                        partIndex: null
                    }
                }
                return null
            }
        }, {
            name: "MediaSequence",
            run: (e,t,i,s,n,r)=>{
                if (!r)
                    return null;
                const a = e.getMediaSequenceMap(r);
                if (!a || 0 === a.size)
                    return null;
                if (void 0 === t.mediaSequence || !Array.isArray(t.segments) || !t.segments.length)
                    return null;
                let o = t.mediaSequence
                  , l = 0;
                for (const h of t.segments) {
                    const e = a.get(o);
                    if (!e)
                        break;
                    if (n >= e.start && n < e.end) {
                        if (Array.isArray(h.parts) && h.parts.length) {
                            let t = e.start
                              , i = 0;
                            for (const s of h.parts) {
                                const r = t
                                  , a = r + s.duration;
                                if (n >= r && n < a)
                                    return {
                                        time: e.start,
                                        segmentIndex: l,
                                        partIndex: i
                                    };
                                i++,
                                t = a
                            }
                        }
                        return {
                            time: e.start,
                            segmentIndex: l,
                            partIndex: null
                        }
                    }
                    l++,
                    o++
                }
                return null
            }
        }, {
            name: "ProgramDateTime",
            run: (e,t,i,s,n)=>{
                if (!Object.keys(e.timelineToDatetimeMappings).length)
                    return null;
                let r = null
                  , a = null;
                const o = Wn(t);
                n = n || 0;
                for (let l = 0; l < o.length; l++) {
                    const i = o[t.endList || 0 === n ? l : o.length - (l + 1)]
                      , s = i.segment
                      , h = e.timelineToDatetimeMappings[s.timeline];
                    if (!h || !s.dateTimeObject)
                        continue;
                    let d = s.dateTimeObject.getTime() / 1e3 + h;
                    if (s.parts && "number" === typeof i.partIndex)
                        for (let e = 0; e < i.partIndex; e++)
                            d += s.parts[e].duration;
                    const u = Math.abs(n - d);
                    if (null !== a && (0 === u || a < u))
                        break;
                    a = u,
                    r = {
                        time: d,
                        segmentIndex: i.segmentIndex,
                        partIndex: i.partIndex
                    }
                }
                return r
            }
        }, {
            name: "Segment",
            run: (e,t,i,s,n)=>{
                let r = null
                  , a = null;
                n = n || 0;
                const o = Wn(t);
                for (let l = 0; l < o.length; l++) {
                    const e = o[t.endList || 0 === n ? l : o.length - (l + 1)]
                      , i = e.segment
                      , h = e.part && e.part.start || i && i.start;
                    if (i.timeline === s && "undefined" !== typeof h) {
                        const t = Math.abs(n - h);
                        if (null !== a && a < t)
                            break;
                        (!r || null === a || a >= t) && (a = t,
                        r = {
                            time: h,
                            segmentIndex: e.segmentIndex,
                            partIndex: e.partIndex
                        })
                    }
                }
                return r
            }
        }, {
            name: "Discontinuity",
            run: (e,t,i,s,n)=>{
                let r = null;
                if (n = n || 0,
                t.discontinuityStarts && t.discontinuityStarts.length) {
                    let i = null;
                    for (let s = 0; s < t.discontinuityStarts.length; s++) {
                        const a = t.discontinuityStarts[s]
                          , o = t.discontinuitySequence + s + 1
                          , l = e.discontinuities[o];
                        if (l) {
                            const e = Math.abs(n - l.time);
                            if (null !== i && i < e)
                                break;
                            (!r || null === i || i >= e) && (i = e,
                            r = {
                                time: l.time,
                                segmentIndex: a,
                                partIndex: null
                            })
                        }
                    }
                }
                return r
            }
        }, {
            name: "Playlist",
            run: (e,t,i,s,n)=>{
                if (t.syncInfo) {
                    return {
                        time: t.syncInfo.time,
                        segmentIndex: t.syncInfo.mediaSequence - t.mediaSequence,
                        partIndex: null
                    }
                }
                return null
            }
        }];
        class _o extends wn.EventTarget {
            constructor(e={}) {
                super(),
                this.timelines = [],
                this.discontinuities = [],
                this.timelineToDatetimeMappings = {},
                this.mediaSequenceStorage_ = new Map,
                this.logger_ = On("SyncController")
            }
            getMediaSequenceMap(e) {
                return this.mediaSequenceStorage_.get(e)
            }
            updateMediaSequenceMap(e, t, i) {
                if (void 0 === e.mediaSequence || !Array.isArray(e.segments) || !e.segments.length)
                    return;
                const s = this.getMediaSequenceMap(i)
                  , n = new Map;
                let r, a = e.mediaSequence;
                s ? s.has(e.mediaSequence) ? r = s.get(e.mediaSequence).start : (this.logger_(`MediaSequence sync for ${i} segment loader - received a gap between playlists.\nFallback base time to: ${t}.\nReceived media sequence: ${a}.\nCurrent map: `, s),
                r = t) : r = 0,
                this.logger_(`MediaSequence sync for ${i} segment loader.\nReceived media sequence: ${a}.\nbase time is ${r}\nCurrent map: `, s),
                e.segments.forEach((e=>{
                    const t = r
                      , i = t + e.duration
                      , s = {
                        start: t,
                        end: i
                    };
                    n.set(a, s),
                    a++,
                    r = i
                }
                )),
                this.mediaSequenceStorage_.set(i, n)
            }
            getSyncPoint(e, t, i, s, n) {
                if (t !== 1 / 0) {
                    return fo.find((({name: e})=>"VOD" === e)).run(this, e, t)
                }
                const r = this.runStrategies_(e, t, i, s, n);
                if (!r.length)
                    return null;
                for (const a of r) {
                    const {syncPoint: t, strategy: i} = a
                      , {segmentIndex: n, time: r} = t;
                    if (n < 0)
                        continue;
                    const o = r
                      , l = o + e.segments[n].duration;
                    if (this.logger_(`Strategy: ${i}. Current time: ${s}. selected segment: ${n}. Time: [${o} -> ${l}]}`),
                    s >= o && s < l)
                        return this.logger_("Found sync point with exact match: ", t),
                        t
                }
                return this.selectSyncPoint_(r, {
                    key: "time",
                    value: s
                })
            }
            getExpiredTime(e, t) {
                if (!e || !e.segments)
                    return null;
                const i = this.runStrategies_(e, t, e.discontinuitySequence, 0, "main");
                if (!i.length)
                    return null;
                const s = this.selectSyncPoint_(i, {
                    key: "segmentIndex",
                    value: 0
                });
                return s.segmentIndex > 0 && (s.time *= -1),
                Math.abs(s.time + Jn({
                    defaultDuration: e.targetDuration,
                    durationList: e.segments,
                    startIndex: s.segmentIndex,
                    endIndex: 0
                }))
            }
            runStrategies_(e, t, i, s, n) {
                const r = [];
                for (let a = 0; a < fo.length; a++) {
                    const o = fo[a]
                      , l = o.run(this, e, t, i, s, n);
                    l && (l.strategy = o.name,
                    r.push({
                        strategy: o.name,
                        syncPoint: l
                    }))
                }
                return r
            }
            selectSyncPoint_(e, t) {
                let i = e[0].syncPoint
                  , s = Math.abs(e[0].syncPoint[t.key] - t.value)
                  , n = e[0].strategy;
                for (let r = 1; r < e.length; r++) {
                    const a = Math.abs(e[r].syncPoint[t.key] - t.value);
                    a < s && (s = a,
                    i = e[r].syncPoint,
                    n = e[r].strategy)
                }
                return this.logger_(`syncPoint for [${t.key}: ${t.value}] chosen with strategy [${n}]: [time:${i.time}, segmentIndex:${i.segmentIndex}` + ("number" === typeof i.partIndex ? `,partIndex:${i.partIndex}` : "") + "]"),
                i
            }
            saveExpiredSegmentInfo(e, t) {
                const i = t.mediaSequence - e.mediaSequence;
                if (i > 86400)
                    wn.log.warn(`Not saving expired segment info. Media sequence gap ${i} is too large.`);
                else
                    for (let s = i - 1; s >= 0; s--) {
                        const i = e.segments[s];
                        if (i && "undefined" !== typeof i.start) {
                            t.syncInfo = {
                                mediaSequence: e.mediaSequence + s,
                                time: i.start
                            },
                            this.logger_(`playlist refresh sync: [time:${t.syncInfo.time}, mediaSequence: ${t.syncInfo.mediaSequence}]`),
                            this.trigger("syncinfoupdate");
                            break
                        }
                    }
            }
            setDateTimeMappingForStart(e) {
                if (this.timelineToDatetimeMappings = {},
                e.segments && e.segments.length && e.segments[0].dateTimeObject) {
                    const t = e.segments[0]
                      , i = t.dateTimeObject.getTime() / 1e3;
                    this.timelineToDatetimeMappings[t.timeline] = -i
                }
            }
            saveSegmentTimingInfo({segmentInfo: e, shouldSaveTimelineMapping: t}) {
                const i = this.calculateSegmentTimeMapping_(e, e.timingInfo, t)
                  , s = e.segment;
                i && (this.saveDiscontinuitySyncInfo_(e),
                e.playlist.syncInfo || (e.playlist.syncInfo = {
                    mediaSequence: e.playlist.mediaSequence + e.mediaIndex,
                    time: s.start
                }));
                const n = s.dateTimeObject;
                s.discontinuity && t && n && (this.timelineToDatetimeMappings[s.timeline] = -n.getTime() / 1e3)
            }
            timestampOffsetForTimeline(e) {
                return "undefined" === typeof this.timelines[e] ? null : this.timelines[e].time
            }
            mappingForTimeline(e) {
                return "undefined" === typeof this.timelines[e] ? null : this.timelines[e].mapping
            }
            calculateSegmentTimeMapping_(e, t, i) {
                const s = e.segment
                  , n = e.part;
                let r, a, o = this.timelines[e.timeline];
                if ("number" === typeof e.timestampOffset)
                    o = {
                        time: e.startOfSegment,
                        mapping: e.startOfSegment - t.start
                    },
                    i && (this.timelines[e.timeline] = o,
                    this.trigger("timestampoffset"),
                    this.logger_(`time mapping for timeline ${e.timeline}: [time: ${o.time}] [mapping: ${o.mapping}]`)),
                    r = e.startOfSegment,
                    a = t.end + o.mapping;
                else {
                    if (!o)
                        return !1;
                    r = t.start + o.mapping,
                    a = t.end + o.mapping
                }
                return n && (n.start = r,
                n.end = a),
                (!s.start || r < s.start) && (s.start = r),
                s.end = a,
                !0
            }
            saveDiscontinuitySyncInfo_(e) {
                const t = e.playlist
                  , i = e.segment;
                if (i.discontinuity)
                    this.discontinuities[i.timeline] = {
                        time: i.start,
                        accuracy: 0
                    };
                else if (t.discontinuityStarts && t.discontinuityStarts.length)
                    for (let s = 0; s < t.discontinuityStarts.length; s++) {
                        const n = t.discontinuityStarts[s]
                          , r = t.discontinuitySequence + s + 1
                          , a = n - e.mediaIndex
                          , o = Math.abs(a);
                        if (!this.discontinuities[r] || this.discontinuities[r].accuracy > o) {
                            let s;
                            s = a < 0 ? i.start - Jn({
                                defaultDuration: t.targetDuration,
                                durationList: t.segments,
                                startIndex: e.mediaIndex,
                                endIndex: n
                            }) : i.end + Jn({
                                defaultDuration: t.targetDuration,
                                durationList: t.segments,
                                startIndex: e.mediaIndex + 1,
                                endIndex: n
                            }),
                            this.discontinuities[r] = {
                                time: s,
                                accuracy: o
                            }
                        }
                    }
            }
            dispose() {
                this.trigger("dispose"),
                this.off()
            }
        }
        class yo extends wn.EventTarget {
            constructor() {
                super(),
                this.pendingTimelineChanges_ = {},
                this.lastTimelineChanges_ = {}
            }
            clearPendingTimelineChange(e) {
                this.pendingTimelineChanges_[e] = null,
                this.trigger("pendingtimelinechange")
            }
            pendingTimelineChange({type: e, from: t, to: i}) {
                return "number" === typeof t && "number" === typeof i && (this.pendingTimelineChanges_[e] = {
                    type: e,
                    from: t,
                    to: i
                },
                this.trigger("pendingtimelinechange")),
                this.pendingTimelineChanges_[e]
            }
            lastTimelineChange({type: e, from: t, to: i}) {
                return "number" === typeof t && "number" === typeof i && (this.lastTimelineChanges_[e] = {
                    type: e,
                    from: t,
                    to: i
                },
                delete this.pendingTimelineChanges_[e],
                this.trigger("timelinechange")),
                this.lastTimelineChanges_[e]
            }
            dispose() {
                this.trigger("dispose"),
                this.pendingTimelineChanges_ = {},
                this.lastTimelineChanges_ = {},
                this.off()
            }
        }
        var vo = Gr(Kr(Qr((function() {
            var e = function() {
                function e() {
                    this.listeners = {}
                }
                var t = e.prototype;
                return t.on = function(e, t) {
                    this.listeners[e] || (this.listeners[e] = []),
                    this.listeners[e].push(t)
                }
                ,
                t.off = function(e, t) {
                    if (!this.listeners[e])
                        return !1;
                    var i = this.listeners[e].indexOf(t);
                    return this.listeners[e] = this.listeners[e].slice(0),
                    this.listeners[e].splice(i, 1),
                    i > -1
                }
                ,
                t.trigger = function(e) {
                    var t = this.listeners[e];
                    if (t)
                        if (2 === arguments.length)
                            for (var i = t.length, s = 0; s < i; ++s)
                                t[s].call(this, arguments[1]);
                        else
                            for (var n = Array.prototype.slice.call(arguments, 1), r = t.length, a = 0; a < r; ++a)
                                t[a].apply(this, n)
                }
                ,
                t.dispose = function() {
                    this.listeners = {}
                }
                ,
                t.pipe = function(e) {
                    this.on("data", (function(t) {
                        e.push(t)
                    }
                    ))
                }
                ,
                e
            }();
            let t = null;
            class s {
                constructor(e) {
                    let i, s, n;
                    t || (t = function() {
                        const e = [[[], [], [], [], []], [[], [], [], [], []]]
                          , t = e[0]
                          , i = e[1]
                          , s = t[4]
                          , n = i[4];
                        let r, a, o;
                        const l = []
                          , h = [];
                        let d, u, c, p, m, g;
                        for (r = 0; r < 256; r++)
                            h[(l[r] = r << 1 ^ 283 * (r >> 7)) ^ r] = r;
                        for (a = o = 0; !s[a]; a ^= d || 1,
                        o = h[o] || 1)
                            for (p = o ^ o << 1 ^ o << 2 ^ o << 3 ^ o << 4,
                            p = p >> 8 ^ 255 & p ^ 99,
                            s[a] = p,
                            n[p] = a,
                            c = l[u = l[d = l[a]]],
                            g = 16843009 * c ^ 65537 * u ^ 257 * d ^ 16843008 * a,
                            m = 257 * l[p] ^ 16843008 * p,
                            r = 0; r < 4; r++)
                                t[r][a] = m = m << 24 ^ m >>> 8,
                                i[r][p] = g = g << 24 ^ g >>> 8;
                        for (r = 0; r < 5; r++)
                            t[r] = t[r].slice(0),
                            i[r] = i[r].slice(0);
                        return e
                    }()),
                    this._tables = [[t[0][0].slice(), t[0][1].slice(), t[0][2].slice(), t[0][3].slice(), t[0][4].slice()], [t[1][0].slice(), t[1][1].slice(), t[1][2].slice(), t[1][3].slice(), t[1][4].slice()]];
                    const r = this._tables[0][4]
                      , a = this._tables[1]
                      , o = e.length;
                    let l = 1;
                    if (4 !== o && 6 !== o && 8 !== o)
                        throw new Error("Invalid aes key size");
                    const h = e.slice(0)
                      , d = [];
                    for (this._key = [h, d],
                    i = o; i < 4 * o + 28; i++)
                        n = h[i - 1],
                        (i % o === 0 || 8 === o && i % o === 4) && (n = r[n >>> 24] << 24 ^ r[n >> 16 & 255] << 16 ^ r[n >> 8 & 255] << 8 ^ r[255 & n],
                        i % o === 0 && (n = n << 8 ^ n >>> 24 ^ l << 24,
                        l = l << 1 ^ 283 * (l >> 7))),
                        h[i] = h[i - o] ^ n;
                    for (s = 0; i; s++,
                    i--)
                        n = h[3 & s ? i : i - 4],
                        d[s] = i <= 4 || s < 4 ? n : a[0][r[n >>> 24]] ^ a[1][r[n >> 16 & 255]] ^ a[2][r[n >> 8 & 255]] ^ a[3][r[255 & n]]
                }
                decrypt(e, t, i, s, n, r) {
                    const a = this._key[1];
                    let o, l, h, d = e ^ a[0], u = s ^ a[1], c = i ^ a[2], p = t ^ a[3];
                    const m = a.length / 4 - 2;
                    let g, f = 4;
                    const _ = this._tables[1]
                      , y = _[0]
                      , v = _[1]
                      , T = _[2]
                      , b = _[3]
                      , S = _[4];
                    for (g = 0; g < m; g++)
                        o = y[d >>> 24] ^ v[u >> 16 & 255] ^ T[c >> 8 & 255] ^ b[255 & p] ^ a[f],
                        l = y[u >>> 24] ^ v[c >> 16 & 255] ^ T[p >> 8 & 255] ^ b[255 & d] ^ a[f + 1],
                        h = y[c >>> 24] ^ v[p >> 16 & 255] ^ T[d >> 8 & 255] ^ b[255 & u] ^ a[f + 2],
                        p = y[p >>> 24] ^ v[d >> 16 & 255] ^ T[u >> 8 & 255] ^ b[255 & c] ^ a[f + 3],
                        f += 4,
                        d = o,
                        u = l,
                        c = h;
                    for (g = 0; g < 4; g++)
                        n[(3 & -g) + r] = S[d >>> 24] << 24 ^ S[u >> 16 & 255] << 16 ^ S[c >> 8 & 255] << 8 ^ S[255 & p] ^ a[f++],
                        o = d,
                        d = u,
                        u = c,
                        c = p,
                        p = o
                }
            }
            class n extends e {
                constructor() {
                    super(e),
                    this.jobs = [],
                    this.delay = 1,
                    this.timeout_ = null
                }
                processJob_() {
                    this.jobs.shift()(),
                    this.jobs.length ? this.timeout_ = setTimeout(this.processJob_.bind(this), this.delay) : this.timeout_ = null
                }
                push(e) {
                    this.jobs.push(e),
                    this.timeout_ || (this.timeout_ = setTimeout(this.processJob_.bind(this), this.delay))
                }
            }
            const r = function(e) {
                return e << 24 | (65280 & e) << 8 | (16711680 & e) >> 8 | e >>> 24
            };
            class a {
                constructor(e, t, i, s) {
                    const o = a.STEP
                      , l = new Int32Array(e.buffer)
                      , h = new Uint8Array(e.byteLength);
                    let d = 0;
                    for (this.asyncStream_ = new n,
                    this.asyncStream_.push(this.decryptChunk_(l.subarray(d, d + o), t, i, h)),
                    d = o; d < l.length; d += o)
                        i = new Uint32Array([r(l[d - 4]), r(l[d - 3]), r(l[d - 2]), r(l[d - 1])]),
                        this.asyncStream_.push(this.decryptChunk_(l.subarray(d, d + o), t, i, h));
                    this.asyncStream_.push((function() {
                        var e;
                        s(null, (e = h).subarray(0, e.byteLength - e[e.byteLength - 1]))
                    }
                    ))
                }
                static get STEP() {
                    return 32e3
                }
                decryptChunk_(e, t, i, n) {
                    return function() {
                        const a = function(e, t, i) {
                            const n = new Int32Array(e.buffer,e.byteOffset,e.byteLength >> 2)
                              , a = new s(Array.prototype.slice.call(t))
                              , o = new Uint8Array(e.byteLength)
                              , l = new Int32Array(o.buffer);
                            let h, d, u, c, p, m, g, f, _;
                            for (h = i[0],
                            d = i[1],
                            u = i[2],
                            c = i[3],
                            _ = 0; _ < n.length; _ += 4)
                                p = r(n[_]),
                                m = r(n[_ + 1]),
                                g = r(n[_ + 2]),
                                f = r(n[_ + 3]),
                                a.decrypt(p, m, g, f, l, _),
                                l[_] = r(l[_] ^ h),
                                l[_ + 1] = r(l[_ + 1] ^ d),
                                l[_ + 2] = r(l[_ + 2] ^ u),
                                l[_ + 3] = r(l[_ + 3] ^ c),
                                h = p,
                                d = m,
                                u = g,
                                c = f;
                            return o
                        }(e, t, i);
                        n.set(a, e.byteOffset)
                    }
                }
            }
            var o, l = "undefined" !== typeof globalThis ? globalThis : "undefined" !== typeof window ? window : "undefined" !== typeof i.g ? i.g : "undefined" !== typeof self ? self : {};
            o = "undefined" !== typeof window ? window : "undefined" !== typeof l ? l : "undefined" !== typeof self ? self : {};
            var h = o.BigInt || Number;
            h("0x1"),
            h("0x100"),
            h("0x10000"),
            h("0x1000000"),
            h("0x100000000"),
            h("0x10000000000"),
            h("0x1000000000000"),
            h("0x100000000000000"),
            h("0x10000000000000000"),
            function() {
                var e = new Uint16Array([65484])
                  , t = new Uint8Array(e.buffer,e.byteOffset,e.byteLength);
                255 === t[0] || t[0]
            }();
            const d = function(e) {
                const t = {};
                return Object.keys(e).forEach((i=>{
                    const s = e[i];
                    var n;
                    n = s,
                    ("function" === ArrayBuffer.isView ? ArrayBuffer.isView(n) : n && n.buffer instanceof ArrayBuffer) ? t[i] = {
                        bytes: s.buffer,
                        byteOffset: s.byteOffset,
                        byteLength: s.byteLength
                    } : t[i] = s
                }
                )),
                t
            };
            self.onmessage = function(e) {
                const t = e.data
                  , i = new Uint8Array(t.encrypted.bytes,t.encrypted.byteOffset,t.encrypted.byteLength)
                  , s = new Uint32Array(t.key.bytes,t.key.byteOffset,t.key.byteLength / 4)
                  , n = new Uint32Array(t.iv.bytes,t.iv.byteOffset,t.iv.byteLength / 4);
                new a(i,s,n,(function(e, i) {
                    self.postMessage(d({
                        source: t.source,
                        decrypted: i
                    }), [i.buffer])
                }
                ))
            }
        }
        ))));
        const To = e=>{
            let t = e.default ? "main" : "alternative";
            return e.characteristics && e.characteristics.indexOf("public.accessibility.describes-video") >= 0 && (t = "main-desc"),
            t
        }
          , bo = (e,t)=>{
            e.abort(),
            e.pause(),
            t && t.activePlaylistLoader && (t.activePlaylistLoader.pause(),
            t.activePlaylistLoader = null)
        }
          , So = (e,t)=>{
            t.activePlaylistLoader = e,
            e.load()
        }
          , ko = {
            AUDIO: (e,t)=>()=>{
                const {mediaTypes: {[e]: i}, excludePlaylist: s} = t
                  , n = i.activeTrack()
                  , r = i.activeGroup()
                  , a = (r.filter((e=>e.default))[0] || r[0]).id
                  , o = i.tracks[a];
                if (n !== o) {
                    wn.log.warn("Problem encountered loading the alternate audio track.Switching back to default.");
                    for (const e in i.tracks)
                        i.tracks[e].enabled = i.tracks[e] === o;
                    i.onTrackChanged()
                } else
                    s({
                        error: {
                            message: "Problem encountered loading the default audio track."
                        }
                    })
            }
            ,
            SUBTITLES: (e,t)=>()=>{
                const {mediaTypes: {[e]: i}} = t;
                wn.log.warn("Problem encountered loading the subtitle track.Disabling subtitle track.");
                const s = i.activeTrack();
                s && (s.mode = "disabled"),
                i.onTrackChanged()
            }
        }
          , Co = {
            AUDIO: (e,t,i)=>{
                if (!t)
                    return;
                const {tech: s, requestOptions: n, segmentLoaders: {[e]: r}} = i;
                t.on("loadedmetadata", (()=>{
                    const e = t.media();
                    r.playlist(e, n),
                    (!s.paused() || e.endList && "none" !== s.preload()) && r.load()
                }
                )),
                t.on("loadedplaylist", (()=>{
                    r.playlist(t.media(), n),
                    s.paused() || r.load()
                }
                )),
                t.on("error", ko[e](e, i))
            }
            ,
            SUBTITLES: (e,t,i)=>{
                const {tech: s, requestOptions: n, segmentLoaders: {[e]: r}, mediaTypes: {[e]: a}} = i;
                t.on("loadedmetadata", (()=>{
                    const e = t.media();
                    r.playlist(e, n),
                    r.track(a.activeTrack()),
                    (!s.paused() || e.endList && "none" !== s.preload()) && r.load()
                }
                )),
                t.on("loadedplaylist", (()=>{
                    r.playlist(t.media(), n),
                    s.paused() || r.load()
                }
                )),
                t.on("error", ko[e](e, i))
            }
        }
          , Eo = {
            AUDIO: (e,t)=>{
                const {vhs: i, sourceType: s, segmentLoaders: {[e]: n}, requestOptions: r, main: {mediaGroups: a}, mediaTypes: {[e]: {groups: o, tracks: l, logger_: h}}, mainPlaylistLoader: d} = t
                  , u = or(d.main);
                a[e] && 0 !== Object.keys(a[e]).length || (a[e] = {
                    main: {
                        default: {
                            default: !0
                        }
                    }
                },
                u && (a[e].main.default.playlists = d.main.playlists));
                for (const c in a[e]) {
                    o[c] || (o[c] = []);
                    for (const n in a[e][c]) {
                        let p, m = a[e][c][n];
                        if (u ? (h(`AUDIO group '${c}' label '${n}' is a main playlist`),
                        m.isMainPlaylist = !0,
                        p = null) : p = "vhs-json" === s && m.playlists ? new kr(m.playlists[0],i,r) : m.resolvedUri ? new kr(m.resolvedUri,i,r) : m.playlists && "dash" === s ? new Vr(m.playlists[0],i,r,d) : null,
                        m = Mn({
                            id: n,
                            playlistLoader: p
                        }, m),
                        Co[e](e, m.playlistLoader, t),
                        o[c].push(m),
                        "undefined" === typeof l[n]) {
                            const e = new wn.AudioTrack({
                                id: n,
                                kind: To(m),
                                enabled: !1,
                                language: m.language,
                                default: m.default,
                                label: n
                            });
                            l[n] = e
                        }
                    }
                }
                n.on("error", ko[e](e, t))
            }
            ,
            SUBTITLES: (e,t)=>{
                const {tech: i, vhs: s, sourceType: n, segmentLoaders: {[e]: r}, requestOptions: a, main: {mediaGroups: o}, mediaTypes: {[e]: {groups: l, tracks: h}}, mainPlaylistLoader: d} = t;
                for (const u in o[e]) {
                    l[u] || (l[u] = []);
                    for (const r in o[e][u]) {
                        if (!s.options_.useForcedSubtitles && o[e][u][r].forced)
                            continue;
                        let c, p = o[e][u][r];
                        if ("hls" === n)
                            c = new kr(p.resolvedUri,s,a);
                        else if ("dash" === n) {
                            if (!p.playlists.filter((e=>e.excludeUntil !== 1 / 0)).length)
                                return;
                            c = new Vr(p.playlists[0],s,a,d)
                        } else
                            "vhs-json" === n && (c = new kr(p.playlists ? p.playlists[0] : p.resolvedUri,s,a));
                        if (p = Mn({
                            id: r,
                            playlistLoader: c
                        }, p),
                        Co[e](e, p.playlistLoader, t),
                        l[u].push(p),
                        "undefined" === typeof h[r]) {
                            const e = i.addRemoteTextTrack({
                                id: r,
                                kind: "subtitles",
                                default: p.default && p.autoselect,
                                language: p.language,
                                label: r
                            }, !1).track;
                            h[r] = e
                        }
                    }
                }
                r.on("error", ko[e](e, t))
            }
            ,
            "CLOSED-CAPTIONS": (e,t)=>{
                const {tech: i, main: {mediaGroups: s}, mediaTypes: {[e]: {groups: n, tracks: r}}} = t;
                for (const a in s[e]) {
                    n[a] || (n[a] = []);
                    for (const t in s[e][a]) {
                        const o = s[e][a][t];
                        if (!/^(?:CC|SERVICE)/.test(o.instreamId))
                            continue;
                        const l = i.options_.vhs && i.options_.vhs.captionServices || {};
                        let h = {
                            label: t,
                            language: o.language,
                            instreamId: o.instreamId,
                            default: o.default && o.autoselect
                        };
                        if (l[h.instreamId] && (h = Mn(h, l[h.instreamId])),
                        void 0 === h.default && delete h.default,
                        n[a].push(Mn({
                            id: t
                        }, o)),
                        "undefined" === typeof r[t]) {
                            const e = i.addRemoteTextTrack({
                                id: h.instreamId,
                                kind: "captions",
                                default: h.default,
                                language: h.language,
                                label: h.label
                            }, !1).track;
                            r[t] = e
                        }
                    }
                }
            }
        }
          , wo = (e,t)=>{
            for (let i = 0; i < e.length; i++) {
                if (rr(t, e[i]))
                    return !0;
                if (e[i].playlists && wo(e[i].playlists, t))
                    return !0
            }
            return !1
        }
          , xo = {
            AUDIO: (e,t)=>()=>{
                const {mediaTypes: {[e]: {tracks: i}}} = t;
                for (const e in i)
                    if (i[e].enabled)
                        return i[e];
                return null
            }
            ,
            SUBTITLES: (e,t)=>()=>{
                const {mediaTypes: {[e]: {tracks: i}}} = t;
                for (const e in i)
                    if ("showing" === i[e].mode || "hidden" === i[e].mode)
                        return i[e];
                return null
            }
        }
          , Io = e=>{
            ["AUDIO", "SUBTITLES", "CLOSED-CAPTIONS"].forEach((t=>{
                Eo[t](t, e)
            }
            ));
            const {mediaTypes: t, mainPlaylistLoader: i, tech: s, vhs: n, segmentLoaders: {AUDIO: r, main: a}} = e;
            ["AUDIO", "SUBTITLES"].forEach((i=>{
                t[i].activeGroup = ((e,t)=>i=>{
                    const {mainPlaylistLoader: s, mediaTypes: {[e]: {groups: n}}} = t
                      , r = s.media();
                    if (!r)
                        return null;
                    let a = null;
                    r.attributes[e] && (a = n[r.attributes[e]]);
                    const o = Object.keys(n);
                    if (!a)
                        if ("AUDIO" === e && o.length > 1 && or(t.main))
                            for (let e = 0; e < o.length; e++) {
                                const t = n[o[e]];
                                if (wo(t, r)) {
                                    a = t;
                                    break
                                }
                            }
                        else
                            n.main ? a = n.main : 1 === o.length && (a = n[o[0]]);
                    return "undefined" === typeof i ? a : null !== i && a && a.filter((e=>e.id === i.id))[0] || null
                }
                )(i, e),
                t[i].activeTrack = xo[i](i, e),
                t[i].onGroupChanged = ((e,t)=>()=>{
                    const {segmentLoaders: {[e]: i, main: s}, mediaTypes: {[e]: n}} = t
                      , r = n.activeTrack()
                      , a = n.getActiveGroup()
                      , o = n.activePlaylistLoader
                      , l = n.lastGroup_;
                    a && l && a.id === l.id || (n.lastGroup_ = a,
                    n.lastTrack_ = r,
                    bo(i, n),
                    a && !a.isMainPlaylist && (a.playlistLoader ? (i.resyncLoader(),
                    So(a.playlistLoader, n)) : o && s.resetEverything()))
                }
                )(i, e),
                t[i].onGroupChanging = ((e,t)=>()=>{
                    const {segmentLoaders: {[e]: i}, mediaTypes: {[e]: s}} = t;
                    s.lastGroup_ = null,
                    i.abort(),
                    i.pause()
                }
                )(i, e),
                t[i].onTrackChanged = ((e,t)=>()=>{
                    const {mainPlaylistLoader: i, segmentLoaders: {[e]: s, main: n}, mediaTypes: {[e]: r}} = t
                      , a = r.activeTrack()
                      , o = r.getActiveGroup()
                      , l = r.activePlaylistLoader
                      , h = r.lastTrack_;
                    if ((!h || !a || h.id !== a.id) && (r.lastGroup_ = o,
                    r.lastTrack_ = a,
                    bo(s, r),
                    o)) {
                        if (o.isMainPlaylist) {
                            if (!a || !h || a.id === h.id)
                                return;
                            const e = t.vhs.playlistController_
                              , s = e.selectPlaylist();
                            if (e.media() === s)
                                return;
                            return r.logger_(`track change. Switching main audio from ${h.id} to ${a.id}`),
                            i.pause(),
                            n.resetEverything(),
                            void e.fastQualityChange_(s)
                        }
                        if ("AUDIO" === e) {
                            if (!o.playlistLoader)
                                return n.setAudio(!0),
                                void n.resetEverything();
                            s.setAudio(!0),
                            n.setAudio(!1)
                        }
                        l !== o.playlistLoader ? (s.track && s.track(a),
                        s.resetEverything(),
                        So(o.playlistLoader, r)) : So(o.playlistLoader, r)
                    }
                }
                )(i, e),
                t[i].getActiveGroup = ((e,{mediaTypes: t})=>()=>{
                    const i = t[e].activeTrack();
                    return i ? t[e].activeGroup(i) : null
                }
                )(i, e)
            }
            ));
            const o = t.AUDIO.activeGroup();
            if (o) {
                const e = (o.filter((e=>e.default))[0] || o[0]).id;
                t.AUDIO.tracks[e].enabled = !0,
                t.AUDIO.onGroupChanged(),
                t.AUDIO.onTrackChanged();
                t.AUDIO.getActiveGroup().playlistLoader ? (a.setAudio(!1),
                r.setAudio(!0)) : a.setAudio(!0)
            }
            i.on("mediachange", (()=>{
                ["AUDIO", "SUBTITLES"].forEach((e=>t[e].onGroupChanged()))
            }
            )),
            i.on("mediachanging", (()=>{
                ["AUDIO", "SUBTITLES"].forEach((e=>t[e].onGroupChanging()))
            }
            ));
            const l = ()=>{
                t.AUDIO.onTrackChanged(),
                s.trigger({
                    type: "usage",
                    name: "vhs-audio-change"
                })
            }
            ;
            s.audioTracks().addEventListener("change", l),
            s.remoteTextTracks().addEventListener("change", t.SUBTITLES.onTrackChanged),
            n.on("dispose", (()=>{
                s.audioTracks().removeEventListener("change", l),
                s.remoteTextTracks().removeEventListener("change", t.SUBTITLES.onTrackChanged)
            }
            )),
            s.clearTracks("audio");
            for (const h in t.AUDIO.tracks)
                s.audioTracks().addTrack(t.AUDIO.tracks[h])
        }
        ;
        class Po {
            constructor() {
                this.priority_ = [],
                this.pathwayClones_ = new Map
            }
            set version(e) {
                1 === e && (this.version_ = e)
            }
            set ttl(e) {
                this.ttl_ = e || 300
            }
            set reloadUri(e) {
                e && (this.reloadUri_ = Ln(this.reloadUri_, e))
            }
            set priority(e) {
                e && e.length && (this.priority_ = e)
            }
            set pathwayClones(e) {
                e && e.length && (this.pathwayClones_ = new Map(e.map((e=>[e.ID, e]))))
            }
            get version() {
                return this.version_
            }
            get ttl() {
                return this.ttl_
            }
            get reloadUri() {
                return this.reloadUri_
            }
            get priority() {
                return this.priority_
            }
            get pathwayClones() {
                return this.pathwayClones_
            }
        }
        class Ao extends wn.EventTarget {
            constructor(e, t) {
                super(),
                this.currentPathway = null,
                this.defaultPathway = null,
                this.queryBeforeStart = !1,
                this.availablePathways_ = new Set,
                this.steeringManifest = new Po,
                this.proxyServerUrl_ = null,
                this.manifestType_ = null,
                this.ttlTimeout_ = null,
                this.request_ = null,
                this.currentPathwayClones = new Map,
                this.nextPathwayClones = new Map,
                this.excludedSteeringManifestURLs = new Set,
                this.logger_ = On("Content Steering"),
                this.xhr_ = e,
                this.getBandwidth_ = t
            }
            assignTagProperties(e, t) {
                this.manifestType_ = t.serverUri ? "HLS" : "DASH";
                const i = t.serverUri || t.serverURL;
                if (!i)
                    return this.logger_(`steering manifest URL is ${i}, cannot request steering manifest.`),
                    void this.trigger("error");
                i.startsWith("data:") ? this.decodeDataUriManifest_(i.substring(i.indexOf(",") + 1)) : (this.steeringManifest.reloadUri = Ln(e, i),
                this.defaultPathway = t.pathwayId || t.defaultServiceLocation,
                this.queryBeforeStart = t.queryBeforeStart,
                this.proxyServerUrl_ = t.proxyServerURL,
                this.defaultPathway && !this.queryBeforeStart && this.trigger("content-steering"))
            }
            requestSteeringManifest(e) {
                const t = this.steeringManifest.reloadUri;
                if (!t)
                    return;
                const i = e ? t : this.getRequestURI(t);
                if (!i)
                    return this.logger_("No valid content steering manifest URIs. Stopping content steering."),
                    this.trigger("error"),
                    void this.dispose();
                this.request_ = this.xhr_({
                    uri: i
                }, ((e,t)=>{
                    if (e) {
                        if (410 === t.status)
                            return this.logger_(`manifest request 410 ${e}.`),
                            this.logger_(`There will be no more content steering requests to ${i} this session.`),
                            void this.excludedSteeringManifestURLs.add(i);
                        if (429 === t.status) {
                            const i = t.responseHeaders["retry-after"];
                            return this.logger_(`manifest request 429 ${e}.`),
                            this.logger_(`content steering will retry in ${i} seconds.`),
                            void this.startTTLTimeout_(parseInt(i, 10))
                        }
                        return this.logger_(`manifest failed to load ${e}.`),
                        void this.startTTLTimeout_()
                    }
                    const s = JSON.parse(this.request_.responseText);
                    this.assignSteeringProperties_(s),
                    this.startTTLTimeout_()
                }
                ))
            }
            setProxyServerUrl_(e) {
                const t = new (n().URL)(e)
                  , i = new (n().URL)(this.proxyServerUrl_);
                return i.searchParams.set("url", encodeURI(t.toString())),
                this.setSteeringParams_(i.toString())
            }
            decodeDataUriManifest_(e) {
                const t = JSON.parse(n().atob(e));
                this.assignSteeringProperties_(t)
            }
            setSteeringParams_(e) {
                const t = new (n().URL)(e)
                  , i = this.getPathway()
                  , s = this.getBandwidth_();
                if (i) {
                    const e = `_${this.manifestType_}_pathway`;
                    t.searchParams.set(e, i)
                }
                if (s) {
                    const e = `_${this.manifestType_}_throughput`;
                    t.searchParams.set(e, s)
                }
                return t.toString()
            }
            assignSteeringProperties_(e) {
                if (this.steeringManifest.version = e.VERSION,
                !this.steeringManifest.version)
                    return this.logger_(`manifest version is ${e.VERSION}, which is not supported.`),
                    void this.trigger("error");
                this.steeringManifest.ttl = e.TTL,
                this.steeringManifest.reloadUri = e["RELOAD-URI"],
                this.steeringManifest.priority = e["PATHWAY-PRIORITY"] || e["SERVICE-LOCATION-PRIORITY"],
                this.steeringManifest.pathwayClones = e["PATHWAY-CLONES"],
                this.nextPathwayClones = this.steeringManifest.pathwayClones,
                this.availablePathways_.size || (this.logger_("There are no available pathways for content steering. Ending content steering."),
                this.trigger("error"),
                this.dispose());
                const t = (e=>{
                    for (const t of e)
                        if (this.availablePathways_.has(t))
                            return t;
                    return [...this.availablePathways_][0]
                }
                )(this.steeringManifest.priority);
                this.currentPathway !== t && (this.currentPathway = t,
                this.trigger("content-steering"))
            }
            getPathway() {
                return this.currentPathway || this.defaultPathway
            }
            getRequestURI(e) {
                if (!e)
                    return null;
                const t = e=>this.excludedSteeringManifestURLs.has(e);
                if (this.proxyServerUrl_) {
                    const i = this.setProxyServerUrl_(e);
                    if (!t(i))
                        return i
                }
                const i = this.setSteeringParams_(e);
                return t(i) ? null : i
            }
            startTTLTimeout_(e=this.steeringManifest.ttl) {
                const t = 1e3 * e;
                this.ttlTimeout_ = n().setTimeout((()=>{
                    this.requestSteeringManifest()
                }
                ), t)
            }
            clearTTLTimeout_() {
                n().clearTimeout(this.ttlTimeout_),
                this.ttlTimeout_ = null
            }
            abort() {
                this.request_ && this.request_.abort(),
                this.request_ = null
            }
            dispose() {
                this.off("content-steering"),
                this.off("error"),
                this.abort(),
                this.clearTTLTimeout_(),
                this.currentPathway = null,
                this.defaultPathway = null,
                this.queryBeforeStart = null,
                this.proxyServerUrl_ = null,
                this.manifestType_ = null,
                this.ttlTimeout_ = null,
                this.request_ = null,
                this.excludedSteeringManifestURLs = new Set,
                this.availablePathways_ = new Set,
                this.steeringManifest = new Po
            }
            addAvailablePathway(e) {
                e && this.availablePathways_.add(e)
            }
            clearAvailablePathways() {
                this.availablePathways_.clear()
            }
            excludePathway(e) {
                return this.availablePathways_.delete(e)
            }
            didDASHTagChange(e, t) {
                return !t && this.steeringManifest.reloadUri || t && (Ln(e, t.serverURL) !== this.steeringManifest.reloadUri || t.defaultServiceLocation !== this.defaultPathway || t.queryBeforeStart !== this.queryBeforeStart || t.proxyServerURL !== this.proxyServerUrl_)
            }
            getAvailablePathways() {
                return this.availablePathways_
            }
        }
        let Lo;
        const Do = ["mediaRequests", "mediaRequestsAborted", "mediaRequestsTimedout", "mediaRequestsErrored", "mediaTransferDuration", "mediaBytesTransferred", "mediaAppends"]
          , Oo = function(e) {
            return this.audioSegmentLoader_[e] + this.mainSegmentLoader_[e]
        };
        class Mo extends wn.EventTarget {
            constructor(e) {
                super();
                const {src: t, withCredentials: i, tech: s, bandwidth: r, externVhs: a, useCueTags: o, playlistExclusionDuration: l, enableLowInitialPlaylist: h, sourceType: d, cacheEncryptionKeys: u, bufferBasedABR: c, leastPixelDiffSelector: p, captionServices: m} = e;
                if (!t)
                    throw new Error("A non-empty playlist URL or JSON manifest string is required");
                let {maxPlaylistRetries: g} = e;
                null !== g && "undefined" !== typeof g || (g = 1 / 0),
                Lo = a,
                this.bufferBasedABR = Boolean(c),
                this.leastPixelDiffSelector = Boolean(p),
                this.withCredentials = i,
                this.tech_ = s,
                this.vhs_ = s.vhs,
                this.sourceType_ = d,
                this.useCueTags_ = o,
                this.playlistExclusionDuration = l,
                this.maxPlaylistRetries = g,
                this.enableLowInitialPlaylist = h,
                this.useCueTags_ && (this.cueTagsTrack_ = this.tech_.addTextTrack("metadata", "ad-cues"),
                this.cueTagsTrack_.inBandMetadataTrackDispatchType = ""),
                this.requestOptions_ = {
                    withCredentials: i,
                    maxPlaylistRetries: g,
                    timeout: null
                },
                this.on("error", this.pauseLoading),
                this.mediaTypes_ = (()=>{
                    const e = {};
                    return ["AUDIO", "SUBTITLES", "CLOSED-CAPTIONS"].forEach((t=>{
                        e[t] = {
                            groups: {},
                            tracks: {},
                            activePlaylistLoader: null,
                            activeGroup: Va,
                            activeTrack: Va,
                            getActiveGroup: Va,
                            onGroupChanged: Va,
                            onTrackChanged: Va,
                            lastTrack_: null,
                            logger_: On(`MediaGroups[${t}]`)
                        }
                    }
                    )),
                    e
                }
                )(),
                this.mediaSource = new (n().MediaSource),
                this.handleDurationChange_ = this.handleDurationChange_.bind(this),
                this.handleSourceOpen_ = this.handleSourceOpen_.bind(this),
                this.handleSourceEnded_ = this.handleSourceEnded_.bind(this),
                this.mediaSource.addEventListener("durationchange", this.handleDurationChange_),
                this.mediaSource.addEventListener("sourceopen", this.handleSourceOpen_),
                this.mediaSource.addEventListener("sourceended", this.handleSourceEnded_),
                this.seekable_ = Rn(),
                this.hasPlayed_ = !1,
                this.syncController_ = new _o(e),
                this.segmentMetadataTrack_ = s.addRemoteTextTrack({
                    kind: "metadata",
                    label: "segment-metadata"
                }, !1).track,
                this.decrypter_ = new vo,
                this.sourceUpdater_ = new ho(this.mediaSource),
                this.inbandTextTracks_ = {},
                this.timelineChangeController_ = new yo,
                this.keyStatusMap_ = new Map;
                const f = {
                    vhs: this.vhs_,
                    parse708captions: e.parse708captions,
                    useDtsForTimestampOffset: e.useDtsForTimestampOffset,
                    captionServices: m,
                    mediaSource: this.mediaSource,
                    currentTime: this.tech_.currentTime.bind(this.tech_),
                    seekable: ()=>this.seekable(),
                    seeking: ()=>this.tech_.seeking(),
                    duration: ()=>this.duration(),
                    hasPlayed: ()=>this.hasPlayed_,
                    goalBufferLength: ()=>this.goalBufferLength(),
                    bandwidth: r,
                    syncController: this.syncController_,
                    decrypter: this.decrypter_,
                    sourceType: this.sourceType_,
                    inbandTextTracks: this.inbandTextTracks_,
                    cacheEncryptionKeys: u,
                    sourceUpdater: this.sourceUpdater_,
                    timelineChangeController: this.timelineChangeController_,
                    exactManifestTimings: e.exactManifestTimings,
                    addMetadataToTextTrack: this.addMetadataToTextTrack.bind(this)
                };
                this.mainPlaylistLoader_ = "dash" === this.sourceType_ ? new Vr(t,this.vhs_,Mn(this.requestOptions_, {
                    addMetadataToTextTrack: this.addMetadataToTextTrack.bind(this)
                })) : new kr(t,this.vhs_,Mn(this.requestOptions_, {
                    addDateRangesToTextTrack: this.addDateRangesToTextTrack_.bind(this)
                })),
                this.setupMainPlaylistLoaderListeners_(),
                this.mainSegmentLoader_ = new Ha(Mn(f, {
                    segmentMetadataTrack: this.segmentMetadataTrack_,
                    loaderType: "main"
                }),e),
                this.audioSegmentLoader_ = new Ha(Mn(f, {
                    loaderType: "audio"
                }),e),
                this.subtitleSegmentLoader_ = new mo(Mn(f, {
                    loaderType: "vtt",
                    featuresNativeTextTracks: this.tech_.featuresNativeTextTracks,
                    loadVttJs: ()=>new Promise(((e,t)=>{
                        function i() {
                            s.off("vttjserror", n),
                            e()
                        }
                        function n() {
                            s.off("vttjsloaded", i),
                            t()
                        }
                        s.one("vttjsloaded", i),
                        s.one("vttjserror", n),
                        s.addWebVttScript_()
                    }
                    ))
                }),e);
                this.contentSteeringController_ = new Ao(this.vhs_.xhr,(()=>this.mainSegmentLoader_.bandwidth)),
                this.setupSegmentLoaderListeners_(),
                this.bufferBasedABR && (this.mainPlaylistLoader_.one("loadedplaylist", (()=>this.startABRTimer_())),
                this.tech_.on("pause", (()=>this.stopABRTimer_())),
                this.tech_.on("play", (()=>this.startABRTimer_()))),
                Do.forEach((e=>{
                    this[e + "_"] = Oo.bind(this, e)
                }
                )),
                this.logger_ = On("pc"),
                this.triggeredFmp4Usage = !1,
                "none" === this.tech_.preload() ? (this.loadOnPlay_ = ()=>{
                    this.loadOnPlay_ = null,
                    this.mainPlaylistLoader_.load()
                }
                ,
                this.tech_.one("play", this.loadOnPlay_)) : this.mainPlaylistLoader_.load(),
                this.timeToLoadedData__ = -1,
                this.mainAppendsToLoadedData__ = -1,
                this.audioAppendsToLoadedData__ = -1;
                const _ = "none" === this.tech_.preload() ? "play" : "loadstart";
                this.tech_.one(_, (()=>{
                    const e = Date.now();
                    this.tech_.one("loadeddata", (()=>{
                        this.timeToLoadedData__ = Date.now() - e,
                        this.mainAppendsToLoadedData__ = this.mainSegmentLoader_.mediaAppends,
                        this.audioAppendsToLoadedData__ = this.audioSegmentLoader_.mediaAppends
                    }
                    ))
                }
                ))
            }
            mainAppendsToLoadedData_() {
                return this.mainAppendsToLoadedData__
            }
            audioAppendsToLoadedData_() {
                return this.audioAppendsToLoadedData__
            }
            appendsToLoadedData_() {
                const e = this.mainAppendsToLoadedData_()
                  , t = this.audioAppendsToLoadedData_();
                return -1 === e || -1 === t ? -1 : e + t
            }
            timeToLoadedData_() {
                return this.timeToLoadedData__
            }
            checkABR_(e="abr") {
                const t = this.selectPlaylist();
                t && this.shouldSwitchToMedia_(t) && this.switchMedia_(t, e)
            }
            switchMedia_(e, t, i) {
                const s = this.media()
                  , n = s && (s.id || s.uri)
                  , r = e && (e.id || e.uri);
                n && n !== r && (this.logger_(`switch media ${n} -> ${r} from ${t}`),
                this.tech_.trigger({
                    type: "usage",
                    name: `vhs-rendition-change-${t}`
                })),
                this.mainPlaylistLoader_.media(e, i)
            }
            switchMediaForDASHContentSteering_() {
                ["AUDIO", "SUBTITLES", "CLOSED-CAPTIONS"].forEach((e=>{
                    const t = this.mediaTypes_[e]
                      , i = t ? t.activeGroup() : null
                      , s = this.contentSteeringController_.getPathway();
                    if (i && s) {
                        const t = (i.length ? i[0].playlists : i.playlists).filter((e=>e.attributes.serviceLocation === s));
                        t.length && this.mediaTypes_[e].activePlaylistLoader.media(t[0])
                    }
                }
                ))
            }
            startABRTimer_() {
                this.stopABRTimer_(),
                this.abrTimer_ = n().setInterval((()=>this.checkABR_()), 250)
            }
            stopABRTimer_() {
                this.tech_.scrubbing && this.tech_.scrubbing() || (n().clearInterval(this.abrTimer_),
                this.abrTimer_ = null)
            }
            getAudioTrackPlaylists_() {
                const e = this.main()
                  , t = e && e.playlists || [];
                if (!e || !e.mediaGroups || !e.mediaGroups.AUDIO)
                    return t;
                const i = e.mediaGroups.AUDIO
                  , s = Object.keys(i);
                let n;
                if (Object.keys(this.mediaTypes_.AUDIO.groups).length)
                    n = this.mediaTypes_.AUDIO.activeTrack();
                else {
                    const e = i.main || s.length && i[s[0]];
                    for (const t in e)
                        if (e[t].default) {
                            n = {
                                label: t
                            };
                            break
                        }
                }
                if (!n)
                    return t;
                const r = [];
                for (const a in i)
                    if (i[a][n.label]) {
                        const t = i[a][n.label];
                        if (t.playlists && t.playlists.length)
                            r.push.apply(r, t.playlists);
                        else if (t.uri)
                            r.push(t);
                        else if (e.playlists.length)
                            for (let i = 0; i < e.playlists.length; i++) {
                                const t = e.playlists[i];
                                t.attributes && t.attributes.AUDIO && t.attributes.AUDIO === a && r.push(t)
                            }
                    }
                return r.length ? r : t
            }
            setupMainPlaylistLoaderListeners_() {
                this.mainPlaylistLoader_.on("loadedmetadata", (()=>{
                    const e = this.mainPlaylistLoader_.media()
                      , t = 1.5 * e.targetDuration * 1e3;
                    nr(this.mainPlaylistLoader_.main, this.mainPlaylistLoader_.media()) ? this.requestOptions_.timeout = 0 : this.requestOptions_.timeout = t,
                    e.endList && "none" !== this.tech_.preload() && (this.mainSegmentLoader_.playlist(e, this.requestOptions_),
                    this.mainSegmentLoader_.load()),
                    Io({
                        sourceType: this.sourceType_,
                        segmentLoaders: {
                            AUDIO: this.audioSegmentLoader_,
                            SUBTITLES: this.subtitleSegmentLoader_,
                            main: this.mainSegmentLoader_
                        },
                        tech: this.tech_,
                        requestOptions: this.requestOptions_,
                        mainPlaylistLoader: this.mainPlaylistLoader_,
                        vhs: this.vhs_,
                        main: this.main(),
                        mediaTypes: this.mediaTypes_,
                        excludePlaylist: this.excludePlaylist.bind(this)
                    }),
                    this.triggerPresenceUsage_(this.main(), e),
                    this.setupFirstPlay(),
                    !this.mediaTypes_.AUDIO.activePlaylistLoader || this.mediaTypes_.AUDIO.activePlaylistLoader.media() ? this.trigger("selectedinitialmedia") : this.mediaTypes_.AUDIO.activePlaylistLoader.one("loadedmetadata", (()=>{
                        this.trigger("selectedinitialmedia")
                    }
                    ))
                }
                )),
                this.mainPlaylistLoader_.on("loadedplaylist", (()=>{
                    this.loadOnPlay_ && this.tech_.off("play", this.loadOnPlay_);
                    let e = this.mainPlaylistLoader_.media();
                    if (!e) {
                        let t;
                        if (this.attachContentSteeringListeners_(),
                        this.initContentSteeringController_(),
                        this.excludeUnsupportedVariants_(),
                        this.enableLowInitialPlaylist && (t = this.selectInitialPlaylist()),
                        t || (t = this.selectPlaylist()),
                        !t || !this.shouldSwitchToMedia_(t))
                            return;
                        this.initialMedia_ = t,
                        this.switchMedia_(this.initialMedia_, "initial");
                        if (!("vhs-json" === this.sourceType_ && this.initialMedia_.segments))
                            return;
                        e = this.initialMedia_
                    }
                    this.handleUpdatedMediaPlaylist(e)
                }
                )),
                this.mainPlaylistLoader_.on("error", (()=>{
                    const e = this.mainPlaylistLoader_.error;
                    this.excludePlaylist({
                        playlistToExclude: e.playlist,
                        error: e
                    })
                }
                )),
                this.mainPlaylistLoader_.on("mediachanging", (()=>{
                    this.mainSegmentLoader_.abort(),
                    this.mainSegmentLoader_.pause()
                }
                )),
                this.mainPlaylistLoader_.on("mediachange", (()=>{
                    const e = this.mainPlaylistLoader_.media()
                      , t = 1.5 * e.targetDuration * 1e3;
                    nr(this.mainPlaylistLoader_.main, this.mainPlaylistLoader_.media()) ? this.requestOptions_.timeout = 0 : this.requestOptions_.timeout = t,
                    "dash" === this.sourceType_ && this.mainPlaylistLoader_.load(),
                    this.mainSegmentLoader_.pause(),
                    this.mainSegmentLoader_.playlist(e, this.requestOptions_),
                    this.waitingForFastQualityPlaylistReceived_ ? this.runFastQualitySwitch_() : this.mainSegmentLoader_.load(),
                    this.tech_.trigger({
                        type: "mediachange",
                        bubbles: !0
                    })
                }
                )),
                this.mainPlaylistLoader_.on("playlistunchanged", (()=>{
                    const e = this.mainPlaylistLoader_.media();
                    if ("playlist-unchanged" === e.lastExcludeReason_)
                        return;
                    this.stuckAtPlaylistEnd_(e) && (this.excludePlaylist({
                        error: {
                            message: "Playlist no longer updating.",
                            reason: "playlist-unchanged"
                        }
                    }),
                    this.tech_.trigger("playliststuck"))
                }
                )),
                this.mainPlaylistLoader_.on("renditiondisabled", (()=>{
                    this.tech_.trigger({
                        type: "usage",
                        name: "vhs-rendition-disabled"
                    })
                }
                )),
                this.mainPlaylistLoader_.on("renditionenabled", (()=>{
                    this.tech_.trigger({
                        type: "usage",
                        name: "vhs-rendition-enabled"
                    })
                }
                ))
            }
            handleUpdatedMediaPlaylist(e) {
                this.useCueTags_ && this.updateAdCues_(e),
                this.mainSegmentLoader_.pause(),
                this.mainSegmentLoader_.playlist(e, this.requestOptions_),
                this.waitingForFastQualityPlaylistReceived_ && this.runFastQualitySwitch_(),
                this.updateDuration(!e.endList),
                this.tech_.paused() || (this.mainSegmentLoader_.load(),
                this.audioSegmentLoader_ && this.audioSegmentLoader_.load())
            }
            triggerPresenceUsage_(e, t) {
                const i = e.mediaGroups || {};
                let s = !0;
                const n = Object.keys(i.AUDIO);
                for (const r in i.AUDIO)
                    for (const e in i.AUDIO[r]) {
                        i.AUDIO[r][e].uri || (s = !1)
                    }
                s && this.tech_.trigger({
                    type: "usage",
                    name: "vhs-demuxed"
                }),
                Object.keys(i.SUBTITLES).length && this.tech_.trigger({
                    type: "usage",
                    name: "vhs-webvtt"
                }),
                Lo.Playlist.isAes(t) && this.tech_.trigger({
                    type: "usage",
                    name: "vhs-aes"
                }),
                n.length && Object.keys(i.AUDIO[n[0]]).length > 1 && this.tech_.trigger({
                    type: "usage",
                    name: "vhs-alternate-audio"
                }),
                this.useCueTags_ && this.tech_.trigger({
                    type: "usage",
                    name: "vhs-playlist-cue-tags"
                })
            }
            shouldSwitchToMedia_(e) {
                const t = this.mainPlaylistLoader_.media() || this.mainPlaylistLoader_.pendingMedia_
                  , i = this.tech_.currentTime()
                  , s = this.bufferLowWaterLine()
                  , n = this.bufferHighWaterLine();
                return function({currentPlaylist: e, buffered: t, currentTime: i, nextPlaylist: s, bufferLowWaterLine: n, bufferHighWaterLine: r, duration: a, bufferBasedABR: o, log: l}) {
                    if (!s)
                        return wn.log.warn("We received no playlist to switch to. Please check your stream."),
                        !1;
                    const h = `allowing switch ${e && e.id || "null"} -> ${s.id}`;
                    if (!e)
                        return l(`${h} as current playlist is not set`),
                        !0;
                    if (s.id === e.id)
                        return !1;
                    const d = Boolean(Fn(t, i).length);
                    if (!e.endList)
                        return d || "number" !== typeof e.partTargetDuration ? (l(`${h} as current playlist is live`),
                        !0) : (l(`not ${h} as current playlist is live llhls, but currentTime isn't in buffered.`),
                        !1);
                    const u = Vn(t, i)
                      , c = o ? zr.EXPERIMENTAL_MAX_BUFFER_LOW_WATER_LINE : zr.MAX_BUFFER_LOW_WATER_LINE;
                    if (a < c)
                        return l(`${h} as duration < max low water line (${a} < ${c})`),
                        !0;
                    const p = s.attributes.BANDWIDTH
                      , m = e.attributes.BANDWIDTH;
                    if (p < m && (!o || u < r)) {
                        let e = `${h} as next bandwidth < current bandwidth (${p} < ${m})`;
                        return o && (e += ` and forwardBuffer < bufferHighWaterLine (${u} < ${r})`),
                        l(e),
                        !0
                    }
                    if ((!o || p > m) && u >= n) {
                        let e = `${h} as forwardBuffer >= bufferLowWaterLine (${u} >= ${n})`;
                        return o && (e += ` and next bandwidth > current bandwidth (${p} > ${m})`),
                        l(e),
                        !0
                    }
                    return l(`not ${h} as no switching criteria met`),
                    !1
                }({
                    buffered: this.tech_.buffered(),
                    currentTime: i,
                    currentPlaylist: t,
                    nextPlaylist: e,
                    bufferLowWaterLine: s,
                    bufferHighWaterLine: n,
                    duration: this.duration(),
                    bufferBasedABR: this.bufferBasedABR,
                    log: this.logger_
                })
            }
            setupSegmentLoaderListeners_() {
                this.mainSegmentLoader_.on("bandwidthupdate", (()=>{
                    this.checkABR_("bandwidthupdate"),
                    this.tech_.trigger("bandwidthupdate")
                }
                )),
                this.mainSegmentLoader_.on("timeout", (()=>{
                    this.bufferBasedABR && this.mainSegmentLoader_.load()
                }
                )),
                this.bufferBasedABR || this.mainSegmentLoader_.on("progress", (()=>{
                    this.trigger("progress")
                }
                )),
                this.mainSegmentLoader_.on("error", (()=>{
                    const e = this.mainSegmentLoader_.error();
                    this.excludePlaylist({
                        playlistToExclude: e.playlist,
                        error: e
                    })
                }
                )),
                this.mainSegmentLoader_.on("appenderror", (()=>{
                    this.error = this.mainSegmentLoader_.error_,
                    this.trigger("error")
                }
                )),
                this.mainSegmentLoader_.on("syncinfoupdate", (()=>{
                    this.onSyncInfoUpdate_()
                }
                )),
                this.mainSegmentLoader_.on("timestampoffset", (()=>{
                    this.tech_.trigger({
                        type: "usage",
                        name: "vhs-timestamp-offset"
                    })
                }
                )),
                this.audioSegmentLoader_.on("syncinfoupdate", (()=>{
                    this.onSyncInfoUpdate_()
                }
                )),
                this.audioSegmentLoader_.on("appenderror", (()=>{
                    this.error = this.audioSegmentLoader_.error_,
                    this.trigger("error")
                }
                )),
                this.mainSegmentLoader_.on("ended", (()=>{
                    this.logger_("main segment loader ended"),
                    this.onEndOfStream()
                }
                )),
                this.mainSegmentLoader_.on("earlyabort", (e=>{
                    this.bufferBasedABR || (this.delegateLoaders_("all", ["abort"]),
                    this.excludePlaylist({
                        error: {
                            message: "Aborted early because there isn't enough bandwidth to complete the request without rebuffering."
                        },
                        playlistExclusionDuration: 10
                    }))
                }
                ));
                const e = ()=>{
                    if (!this.sourceUpdater_.hasCreatedSourceBuffers())
                        return this.tryToCreateSourceBuffers_();
                    const e = this.getCodecsOrExclude_();
                    e && this.sourceUpdater_.addOrChangeSourceBuffers(e)
                }
                ;
                this.mainSegmentLoader_.on("trackinfo", e),
                this.audioSegmentLoader_.on("trackinfo", e),
                this.mainSegmentLoader_.on("fmp4", (()=>{
                    this.triggeredFmp4Usage || (this.tech_.trigger({
                        type: "usage",
                        name: "vhs-fmp4"
                    }),
                    this.triggeredFmp4Usage = !0)
                }
                )),
                this.audioSegmentLoader_.on("fmp4", (()=>{
                    this.triggeredFmp4Usage || (this.tech_.trigger({
                        type: "usage",
                        name: "vhs-fmp4"
                    }),
                    this.triggeredFmp4Usage = !0)
                }
                )),
                this.audioSegmentLoader_.on("ended", (()=>{
                    this.logger_("audioSegmentLoader ended"),
                    this.onEndOfStream()
                }
                ))
            }
            mediaSecondsLoaded_() {
                return Math.max(this.audioSegmentLoader_.mediaSecondsLoaded + this.mainSegmentLoader_.mediaSecondsLoaded)
            }
            load() {
                this.mainSegmentLoader_.load(),
                this.mediaTypes_.AUDIO.activePlaylistLoader && this.audioSegmentLoader_.load(),
                this.mediaTypes_.SUBTITLES.activePlaylistLoader && this.subtitleSegmentLoader_.load()
            }
            fastQualityChange_(e=this.selectPlaylist()) {
                e && e === this.mainPlaylistLoader_.media() ? this.logger_("skipping fastQualityChange because new media is same as old") : (this.switchMedia_(e, "fast-quality"),
                this.waitingForFastQualityPlaylistReceived_ = !0)
            }
            runFastQualitySwitch_() {
                this.waitingForFastQualityPlaylistReceived_ = !1,
                this.mainSegmentLoader_.pause(),
                this.mainSegmentLoader_.resetEverything((()=>{
                    this.tech_.setCurrentTime(this.tech_.currentTime())
                }
                ))
            }
            play() {
                if (this.setupFirstPlay())
                    return;
                this.tech_.ended() && this.tech_.setCurrentTime(0),
                this.hasPlayed_ && this.load();
                const e = this.tech_.seekable();
                return this.tech_.duration() === 1 / 0 && this.tech_.currentTime() < e.start(0) ? this.tech_.setCurrentTime(e.end(e.length - 1)) : void 0
            }
            setupFirstPlay() {
                const e = this.mainPlaylistLoader_.media();
                if (!e || this.tech_.paused() || this.hasPlayed_)
                    return !1;
                if (!e.endList || e.start) {
                    const t = this.seekable();
                    if (!t.length)
                        return !1;
                    const i = t.end(0);
                    let s = i;
                    if (e.start) {
                        const n = e.start.timeOffset;
                        s = n < 0 ? Math.max(i + n, t.start(0)) : Math.min(i, n)
                    }
                    this.trigger("firstplay"),
                    this.tech_.setCurrentTime(s)
                }
                return this.hasPlayed_ = !0,
                this.load(),
                !0
            }
            handleSourceOpen_() {
                if (this.tryToCreateSourceBuffers_(),
                this.tech_.autoplay()) {
                    const e = this.tech_.play();
                    "undefined" !== typeof e && "function" === typeof e.then && e.then(null, (e=>{}
                    ))
                }
                this.trigger("sourceopen")
            }
            handleSourceEnded_() {
                if (!this.inbandTextTracks_.metadataTrack_)
                    return;
                const e = this.inbandTextTracks_.metadataTrack_.cues;
                if (!e || !e.length)
                    return;
                const t = this.duration();
                e[e.length - 1].endTime = isNaN(t) || Math.abs(t) === 1 / 0 ? Number.MAX_VALUE : t
            }
            handleDurationChange_() {
                this.tech_.trigger("durationchange")
            }
            onEndOfStream() {
                let e = this.mainSegmentLoader_.ended_;
                if (this.mediaTypes_.AUDIO.activePlaylistLoader) {
                    const t = this.mainSegmentLoader_.getCurrentMediaInfo_();
                    e = !t || t.hasVideo ? e && this.audioSegmentLoader_.ended_ : this.audioSegmentLoader_.ended_
                }
                e && (this.stopABRTimer_(),
                this.sourceUpdater_.endOfStream())
            }
            stuckAtPlaylistEnd_(e) {
                if (!this.seekable().length)
                    return !1;
                const t = this.syncController_.getExpiredTime(e, this.duration());
                if (null === t)
                    return !1;
                const i = Lo.Playlist.playlistEnd(e, t)
                  , s = this.tech_.currentTime()
                  , n = this.tech_.buffered();
                if (!n.length)
                    return i - s <= Bn;
                const r = n.end(n.length - 1);
                return r - s <= Bn && i - r <= Bn
            }
            excludePlaylist({playlistToExclude: e=this.mainPlaylistLoader_.media(), error: t={}, playlistExclusionDuration: i}) {
                if (e = e || this.mainPlaylistLoader_.media(),
                i = i || t.playlistExclusionDuration || this.playlistExclusionDuration,
                !e)
                    return this.error = t,
                    void ("open" !== this.mediaSource.readyState ? this.trigger("error") : this.sourceUpdater_.endOfStream("network"));
                e.playlistErrors_++;
                const s = this.mainPlaylistLoader_.main.playlists
                  , n = s.filter(ir)
                  , r = 1 === n.length && n[0] === e;
                if (1 === s.length && i !== 1 / 0)
                    return wn.log.warn(`Problem encountered with playlist ${e.id}. Trying again since it is the only playlist.`),
                    this.tech_.trigger("retryplaylist"),
                    this.mainPlaylistLoader_.load(r);
                if (r) {
                    if (this.main().contentSteering) {
                        const t = this.pathwayAttribute_(e)
                          , i = 1e3 * this.contentSteeringController_.steeringManifest.ttl;
                        return this.contentSteeringController_.excludePathway(t),
                        this.excludeThenChangePathway_(),
                        void setTimeout((()=>{
                            this.contentSteeringController_.addAvailablePathway(t)
                        }
                        ), i)
                    }
                    let t = !1;
                    s.forEach((i=>{
                        if (i === e)
                            return;
                        const s = i.excludeUntil;
                        "undefined" !== typeof s && s !== 1 / 0 && (t = !0,
                        delete i.excludeUntil)
                    }
                    )),
                    t && (wn.log.warn("Removing other playlists from the exclusion list because the last rendition is about to be excluded."),
                    this.tech_.trigger("retryplaylist"))
                }
                let a;
                a = e.playlistErrors_ > this.maxPlaylistRetries ? 1 / 0 : Date.now() + 1e3 * i,
                e.excludeUntil = a,
                t.reason && (e.lastExcludeReason_ = t.reason),
                this.tech_.trigger("excludeplaylist"),
                this.tech_.trigger({
                    type: "usage",
                    name: "vhs-rendition-excluded"
                });
                const o = this.selectPlaylist();
                if (!o)
                    return this.error = "Playback cannot continue. No available working or supported playlists.",
                    void this.trigger("error");
                const l = t.internal ? this.logger_ : wn.log.warn
                  , h = t.message ? " " + t.message : "";
                l(`${t.internal ? "Internal problem" : "Problem"} encountered with playlist ${e.id}.${h} Switching to playlist ${o.id}.`),
                o.attributes.AUDIO !== e.attributes.AUDIO && this.delegateLoaders_("audio", ["abort", "pause"]),
                o.attributes.SUBTITLES !== e.attributes.SUBTITLES && this.delegateLoaders_("subtitle", ["abort", "pause"]),
                this.delegateLoaders_("main", ["abort", "pause"]);
                const d = o.targetDuration / 2 * 1e3 || 5e3
                  , u = "number" === typeof o.lastRequest && Date.now() - o.lastRequest <= d;
                return this.switchMedia_(o, "exclude", r || u)
            }
            pauseLoading() {
                this.delegateLoaders_("all", ["abort", "pause"]),
                this.stopABRTimer_()
            }
            delegateLoaders_(e, t) {
                const i = []
                  , s = "all" === e;
                (s || "main" === e) && i.push(this.mainPlaylistLoader_);
                const n = [];
                (s || "audio" === e) && n.push("AUDIO"),
                (s || "subtitle" === e) && (n.push("CLOSED-CAPTIONS"),
                n.push("SUBTITLES")),
                n.forEach((e=>{
                    const t = this.mediaTypes_[e] && this.mediaTypes_[e].activePlaylistLoader;
                    t && i.push(t)
                }
                )),
                ["main", "audio", "subtitle"].forEach((t=>{
                    const s = this[`${t}SegmentLoader_`];
                    !s || e !== t && "all" !== e || i.push(s)
                }
                )),
                i.forEach((e=>t.forEach((t=>{
                    "function" === typeof e[t] && e[t]()
                }
                ))))
            }
            setCurrentTime(e) {
                const t = Fn(this.tech_.buffered(), e);
                return this.mainPlaylistLoader_ && this.mainPlaylistLoader_.media() && this.mainPlaylistLoader_.media().segments ? t && t.length ? e : (this.mainSegmentLoader_.pause(),
                this.mainSegmentLoader_.resetEverything(),
                this.mediaTypes_.AUDIO.activePlaylistLoader && (this.audioSegmentLoader_.pause(),
                this.audioSegmentLoader_.resetEverything()),
                this.mediaTypes_.SUBTITLES.activePlaylistLoader && (this.subtitleSegmentLoader_.pause(),
                this.subtitleSegmentLoader_.resetEverything()),
                void this.load()) : 0
            }
            duration() {
                if (!this.mainPlaylistLoader_)
                    return 0;
                const e = this.mainPlaylistLoader_.media();
                return e ? e.endList ? this.mediaSource ? this.mediaSource.duration : Lo.Playlist.duration(e) : 1 / 0 : 0
            }
            seekable() {
                return this.seekable_
            }
            onSyncInfoUpdate_() {
                let e;
                if (!this.mainPlaylistLoader_)
                    return;
                let t = this.mainPlaylistLoader_.media();
                if (!t)
                    return;
                let i = this.syncController_.getExpiredTime(t, this.duration());
                if (null === i)
                    return;
                const s = this.mainPlaylistLoader_.main
                  , n = Lo.Playlist.seekable(t, i, Lo.Playlist.liveEdgeDelay(s, t));
                if (0 === n.length)
                    return;
                if (this.mediaTypes_.AUDIO.activePlaylistLoader) {
                    if (t = this.mediaTypes_.AUDIO.activePlaylistLoader.media(),
                    i = this.syncController_.getExpiredTime(t, this.duration()),
                    null === i)
                        return;
                    if (e = Lo.Playlist.seekable(t, i, Lo.Playlist.liveEdgeDelay(s, t)),
                    0 === e.length)
                        return
                }
                let r, a;
                this.seekable_ && this.seekable_.length && (r = this.seekable_.end(0),
                a = this.seekable_.start(0)),
                e ? e.start(0) > n.end(0) || n.start(0) > e.end(0) ? this.seekable_ = n : this.seekable_ = Rn([[e.start(0) > n.start(0) ? e.start(0) : n.start(0), e.end(0) < n.end(0) ? e.end(0) : n.end(0)]]) : this.seekable_ = n,
                this.seekable_ && this.seekable_.length && this.seekable_.end(0) === r && this.seekable_.start(0) === a || (this.logger_(`seekable updated [${$n(this.seekable_)}]`),
                this.tech_.trigger("seekablechanged"))
            }
            updateDuration(e) {
                if (this.updateDuration_ && (this.mediaSource.removeEventListener("sourceopen", this.updateDuration_),
                this.updateDuration_ = null),
                "open" !== this.mediaSource.readyState)
                    return this.updateDuration_ = this.updateDuration.bind(this, e),
                    void this.mediaSource.addEventListener("sourceopen", this.updateDuration_);
                if (e) {
                    const e = this.seekable();
                    if (!e.length)
                        return;
                    return void ((isNaN(this.mediaSource.duration) || this.mediaSource.duration < e.end(e.length - 1)) && this.sourceUpdater_.setDuration(e.end(e.length - 1)))
                }
                const t = this.tech_.buffered();
                let i = Lo.Playlist.duration(this.mainPlaylistLoader_.media());
                t.length > 0 && (i = Math.max(i, t.end(t.length - 1))),
                this.mediaSource.duration !== i && this.sourceUpdater_.setDuration(i)
            }
            dispose() {
                this.trigger("dispose"),
                this.decrypter_.terminate(),
                this.mainPlaylistLoader_.dispose(),
                this.mainSegmentLoader_.dispose(),
                this.contentSteeringController_.dispose(),
                this.keyStatusMap_.clear(),
                this.loadOnPlay_ && this.tech_.off("play", this.loadOnPlay_),
                ["AUDIO", "SUBTITLES"].forEach((e=>{
                    const t = this.mediaTypes_[e].groups;
                    for (const i in t)
                        t[i].forEach((e=>{
                            e.playlistLoader && e.playlistLoader.dispose()
                        }
                        ))
                }
                )),
                this.audioSegmentLoader_.dispose(),
                this.subtitleSegmentLoader_.dispose(),
                this.sourceUpdater_.dispose(),
                this.timelineChangeController_.dispose(),
                this.stopABRTimer_(),
                this.updateDuration_ && this.mediaSource.removeEventListener("sourceopen", this.updateDuration_),
                this.mediaSource.removeEventListener("durationchange", this.handleDurationChange_),
                this.mediaSource.removeEventListener("sourceopen", this.handleSourceOpen_),
                this.mediaSource.removeEventListener("sourceended", this.handleSourceEnded_),
                this.off()
            }
            main() {
                return this.mainPlaylistLoader_.main
            }
            media() {
                return this.mainPlaylistLoader_.media() || this.initialMedia_
            }
            areMediaTypesKnown_() {
                const e = !!this.mediaTypes_.AUDIO.activePlaylistLoader
                  , t = !!this.mainSegmentLoader_.getCurrentMediaInfo_()
                  , i = !e || !!this.audioSegmentLoader_.getCurrentMediaInfo_();
                return !(!t || !i)
            }
            getCodecsOrExclude_() {
                const e = {
                    main: this.mainSegmentLoader_.getCurrentMediaInfo_() || {},
                    audio: this.audioSegmentLoader_.getCurrentMediaInfo_() || {}
                }
                  , t = this.mainSegmentLoader_.getPendingSegmentPlaylist() || this.media();
                e.video = e.main;
                const i = ka(this.main(), t)
                  , s = {}
                  , n = !!this.mediaTypes_.AUDIO.activePlaylistLoader;
                if (e.main.hasVideo && (s.video = i.video || e.main.videoCodec || y.xz),
                e.main.isMuxed && (s.video += `,${i.audio || e.main.audioCodec || y.lA}`),
                (e.main.hasAudio && !e.main.isMuxed || e.audio.hasAudio || n) && (s.audio = i.audio || e.main.audioCodec || e.audio.audioCodec || y.lA,
                e.audio.isFmp4 = e.main.hasAudio && !e.main.isMuxed ? e.main.isFmp4 : e.audio.isFmp4),
                !s.audio && !s.video)
                    return void this.excludePlaylist({
                        playlistToExclude: t,
                        error: {
                            message: "Could not determine codecs for playlist."
                        },
                        playlistExclusionDuration: 1 / 0
                    });
                const r = {};
                let a;
                if (["video", "audio"].forEach((function(t) {
                    if (s.hasOwnProperty(t) && (i = e[t].isFmp4,
                    n = s[t],
                    !(i ? (0,
                    y.p7)(n) : (0,
                    y.Hi)(n)))) {
                        const i = e[t].isFmp4 ? "browser" : "muxer";
                        r[i] = r[i] || [],
                        r[i].push(s[t]),
                        "audio" === t && (a = i)
                    }
                    var i, n
                }
                )),
                n && a && t.attributes.AUDIO) {
                    const e = t.attributes.AUDIO;
                    this.main().playlists.forEach((i=>{
                        (i.attributes && i.attributes.AUDIO) === e && i !== t && (i.excludeUntil = 1 / 0)
                    }
                    )),
                    this.logger_(`excluding audio group ${e} as ${a} does not support codec(s): "${s.audio}"`)
                }
                if (!Object.keys(r).length) {
                    if (this.sourceUpdater_.hasCreatedSourceBuffers() && !this.sourceUpdater_.canChangeType()) {
                        const e = [];
                        if (["video", "audio"].forEach((t=>{
                            const i = ((0,
                            y.kS)(this.sourceUpdater_.codecs[t] || "")[0] || {}).type
                              , n = ((0,
                            y.kS)(s[t] || "")[0] || {}).type;
                            i && n && i.toLowerCase() !== n.toLowerCase() && e.push(`"${this.sourceUpdater_.codecs[t]}" -> "${s[t]}"`)
                        }
                        )),
                        e.length)
                            return void this.excludePlaylist({
                                playlistToExclude: t,
                                error: {
                                    message: `Codec switching not supported: ${e.join(", ")}.`,
                                    internal: !0
                                },
                                playlistExclusionDuration: 1 / 0
                            })
                    }
                    return s
                }
                {
                    const e = Object.keys(r).reduce(((e,t)=>(e && (e += ", "),
                    e += `${t} does not support codec(s): "${r[t].join(",")}"`)), "") + ".";
                    this.excludePlaylist({
                        playlistToExclude: t,
                        error: {
                            internal: !0,
                            message: e
                        },
                        playlistExclusionDuration: 1 / 0
                    })
                }
            }
            tryToCreateSourceBuffers_() {
                if ("open" !== this.mediaSource.readyState || this.sourceUpdater_.hasCreatedSourceBuffers())
                    return;
                if (!this.areMediaTypesKnown_())
                    return;
                const e = this.getCodecsOrExclude_();
                if (!e)
                    return;
                this.sourceUpdater_.createSourceBuffers(e);
                const t = [e.video, e.audio].filter(Boolean).join(",");
                this.excludeIncompatibleVariants_(t)
            }
            excludeUnsupportedVariants_() {
                const e = this.main().playlists
                  , t = [];
                Object.keys(e).forEach((i=>{
                    const s = e[i];
                    if (-1 !== t.indexOf(s.id))
                        return;
                    t.push(s.id);
                    const n = ka(this.main, s)
                      , r = [];
                    !n.audio || (0,
                    y.Hi)(n.audio) || (0,
                    y.p7)(n.audio) || r.push(`audio codec ${n.audio}`),
                    !n.video || (0,
                    y.Hi)(n.video) || (0,
                    y.p7)(n.video) || r.push(`video codec ${n.video}`),
                    n.text && "stpp.ttml.im1t" === n.text && r.push(`text codec ${n.text}`),
                    r.length && (s.excludeUntil = 1 / 0,
                    this.logger_(`excluding ${s.id} for unsupported: ${r.join(", ")}`))
                }
                ))
            }
            excludeIncompatibleVariants_(e) {
                const t = []
                  , i = this.main().playlists
                  , s = ba((0,
                y.kS)(e))
                  , n = Sa(s)
                  , r = s.video && (0,
                y.kS)(s.video)[0] || null
                  , a = s.audio && (0,
                y.kS)(s.audio)[0] || null;
                Object.keys(i).forEach((e=>{
                    const s = i[e];
                    if (-1 !== t.indexOf(s.id) || s.excludeUntil === 1 / 0)
                        return;
                    t.push(s.id);
                    const o = []
                      , l = ka(this.mainPlaylistLoader_.main, s)
                      , h = Sa(l);
                    if (l.audio || l.video) {
                        if (h !== n && o.push(`codec count "${h}" !== "${n}"`),
                        !this.sourceUpdater_.canChangeType()) {
                            const e = l.video && (0,
                            y.kS)(l.video)[0] || null
                              , t = l.audio && (0,
                            y.kS)(l.audio)[0] || null;
                            e && r && e.type.toLowerCase() !== r.type.toLowerCase() && o.push(`video codec "${e.type}" !== "${r.type}"`),
                            t && a && t.type.toLowerCase() !== a.type.toLowerCase() && o.push(`audio codec "${t.type}" !== "${a.type}"`)
                        }
                        o.length && (s.excludeUntil = 1 / 0,
                        this.logger_(`excluding ${s.id}: ${o.join(" && ")}`))
                    }
                }
                ))
            }
            updateAdCues_(e) {
                let t = 0;
                const i = this.seekable();
                i.length && (t = i.start(0)),
                function(e, t, i=0) {
                    if (!e.segments)
                        return;
                    let s, r = i;
                    for (let a = 0; a < e.segments.length; a++) {
                        const i = e.segments[a];
                        if (s || (s = go(t, r + i.duration / 2)),
                        s) {
                            if ("cueIn"in i) {
                                s.endTime = r,
                                s.adEndTime = r,
                                r += i.duration,
                                s = null;
                                continue
                            }
                            if (r < s.endTime) {
                                r += i.duration;
                                continue
                            }
                            s.endTime += i.duration
                        } else if ("cueOut"in i && (s = new (n().VTTCue)(r,r + i.duration,i.cueOut),
                        s.adStartTime = r,
                        s.adEndTime = r + parseFloat(i.cueOut),
                        t.addCue(s)),
                        "cueOutCont"in i) {
                            const [e,a] = i.cueOutCont.split("/").map(parseFloat);
                            s = new (n().VTTCue)(r,r + i.duration,""),
                            s.adStartTime = r - e,
                            s.adEndTime = s.adStartTime + a,
                            t.addCue(s)
                        }
                        r += i.duration
                    }
                }(e, this.cueTagsTrack_, t)
            }
            goalBufferLength() {
                const e = this.tech_.currentTime()
                  , t = zr.GOAL_BUFFER_LENGTH
                  , i = zr.GOAL_BUFFER_LENGTH_RATE
                  , s = Math.max(t, zr.MAX_GOAL_BUFFER_LENGTH);
                return Math.min(t + e * i, s)
            }
            bufferLowWaterLine() {
                const e = this.tech_.currentTime()
                  , t = zr.BUFFER_LOW_WATER_LINE
                  , i = zr.BUFFER_LOW_WATER_LINE_RATE
                  , s = Math.max(t, zr.MAX_BUFFER_LOW_WATER_LINE)
                  , n = Math.max(t, zr.EXPERIMENTAL_MAX_BUFFER_LOW_WATER_LINE);
                return Math.min(t + e * i, this.bufferBasedABR ? n : s)
            }
            bufferHighWaterLine() {
                return zr.BUFFER_HIGH_WATER_LINE
            }
            addDateRangesToTextTrack_(e) {
                Ma(this.inbandTextTracks_, "com.apple.streaming", this.tech_),
                (({inbandTextTracks: e, dateRanges: t})=>{
                    const i = e.metadataTrack_;
                    if (!i)
                        return;
                    const s = n().WebKitDataCue || n().VTTCue;
                    t.forEach((e=>{
                        for (const t of Object.keys(e)) {
                            if (Oa.has(t))
                                continue;
                            const n = new s(e.startTime,e.endTime,"");
                            n.id = e.id,
                            n.type = "com.apple.quicktime.HLS",
                            n.value = {
                                key: Da[t],
                                data: e[t]
                            },
                            "scte35Out" !== t && "scte35In" !== t || (n.value.data = new Uint8Array(n.value.data.match(/[\da-f]{2}/gi)).buffer),
                            i.addCue(n)
                        }
                        e.processDateRange()
                    }
                    ))
                }
                )({
                    inbandTextTracks: this.inbandTextTracks_,
                    dateRanges: e
                })
            }
            addMetadataToTextTrack(e, t, i) {
                const s = this.sourceUpdater_.videoBuffer ? this.sourceUpdater_.videoTimestampOffset() : this.sourceUpdater_.audioTimestampOffset();
                Ma(this.inbandTextTracks_, e, this.tech_),
                La({
                    inbandTextTracks: this.inbandTextTracks_,
                    metadataArray: t,
                    timestampOffset: s,
                    videoDuration: i
                })
            }
            pathwayAttribute_(e) {
                return e.attributes["PATHWAY-ID"] || e.attributes.serviceLocation
            }
            initContentSteeringController_() {
                const e = this.main();
                if (e.contentSteering) {
                    for (const t of e.playlists)
                        this.contentSteeringController_.addAvailablePathway(this.pathwayAttribute_(t));
                    this.contentSteeringController_.assignTagProperties(e.uri, e.contentSteering),
                    this.contentSteeringController_.queryBeforeStart ? this.contentSteeringController_.requestSteeringManifest(!0) : this.tech_.one("canplay", (()=>{
                        this.contentSteeringController_.requestSteeringManifest()
                    }
                    ))
                }
            }
            resetContentSteeringController_() {
                this.contentSteeringController_.clearAvailablePathways(),
                this.contentSteeringController_.dispose(),
                this.initContentSteeringController_()
            }
            attachContentSteeringListeners_() {
                this.contentSteeringController_.on("content-steering", this.excludeThenChangePathway_.bind(this)),
                "dash" === this.sourceType_ && this.mainPlaylistLoader_.on("loadedplaylist", (()=>{
                    const e = this.main();
                    (this.contentSteeringController_.didDASHTagChange(e.uri, e.contentSteering) || (()=>{
                        const t = this.contentSteeringController_.getAvailablePathways()
                          , i = [];
                        for (const s of e.playlists) {
                            const e = s.attributes.serviceLocation;
                            if (e && (i.push(e),
                            !t.has(e)))
                                return !0
                        }
                        return !(i.length || !t.size)
                    }
                    )()) && this.resetContentSteeringController_()
                }
                ))
            }
            excludeThenChangePathway_() {
                const e = this.contentSteeringController_.getPathway();
                if (!e)
                    return;
                this.handlePathwayClones_();
                const t = this.main().playlists
                  , i = new Set;
                let s = !1;
                Object.keys(t).forEach((n=>{
                    const r = t[n]
                      , a = this.pathwayAttribute_(r)
                      , o = a && e !== a;
                    r.excludeUntil === 1 / 0 && "content-steering" === r.lastExcludeReason_ && !o && (delete r.excludeUntil,
                    delete r.lastExcludeReason_,
                    s = !0);
                    const l = !r.excludeUntil && r.excludeUntil !== 1 / 0;
                    !i.has(r.id) && o && l && (i.add(r.id),
                    r.excludeUntil = 1 / 0,
                    r.lastExcludeReason_ = "content-steering",
                    this.logger_(`excluding ${r.id} for ${r.lastExcludeReason_}`))
                }
                )),
                "DASH" === this.contentSteeringController_.manifestType_ && Object.keys(this.mediaTypes_).forEach((t=>{
                    const i = this.mediaTypes_[t];
                    if (i.activePlaylistLoader) {
                        const t = i.activePlaylistLoader.media_;
                        t && t.attributes.serviceLocation !== e && (s = !0)
                    }
                }
                )),
                s && this.changeSegmentPathway_()
            }
            handlePathwayClones_() {
                const e = this.main().playlists
                  , t = this.contentSteeringController_.currentPathwayClones
                  , i = this.contentSteeringController_.nextPathwayClones;
                if (t && t.size || i && i.size) {
                    for (const [e,s] of t.entries()) {
                        i.get(e) || (this.mainPlaylistLoader_.updateOrDeleteClone(s),
                        this.contentSteeringController_.excludePathway(e))
                    }
                    for (const [s,n] of i.entries()) {
                        const i = t.get(s);
                        if (i)
                            this.equalPathwayClones_(i, n) || (this.mainPlaylistLoader_.updateOrDeleteClone(n, !0),
                            this.contentSteeringController_.addAvailablePathway(s));
                        else {
                            e.filter((e=>e.attributes["PATHWAY-ID"] === n["BASE-ID"])).forEach((e=>{
                                this.mainPlaylistLoader_.addClonePathway(n, e)
                            }
                            )),
                            this.contentSteeringController_.addAvailablePathway(s)
                        }
                    }
                    this.contentSteeringController_.currentPathwayClones = new Map(JSON.parse(JSON.stringify([...i])))
                }
            }
            equalPathwayClones_(e, t) {
                if (e["BASE-ID"] !== t["BASE-ID"] || e.ID !== t.ID || e["URI-REPLACEMENT"].HOST !== t["URI-REPLACEMENT"].HOST)
                    return !1;
                const i = e["URI-REPLACEMENT"].PARAMS
                  , s = t["URI-REPLACEMENT"].PARAMS;
                for (const n in i)
                    if (i[n] !== s[n])
                        return !1;
                for (const n in s)
                    if (i[n] !== s[n])
                        return !1;
                return !0
            }
            changeSegmentPathway_() {
                const e = this.selectPlaylist();
                this.pauseLoading(),
                "DASH" === this.contentSteeringController_.manifestType_ && this.switchMediaForDASHContentSteering_(),
                this.switchMedia_(e, "content-steering")
            }
            excludeNonUsablePlaylistsByKeyId_() {
                if (!this.mainPlaylistLoader_ || !this.mainPlaylistLoader_.main)
                    return;
                let e = 0;
                const t = "non-usable";
                this.mainPlaylistLoader_.main.playlists.forEach((i=>{
                    const s = this.mainPlaylistLoader_.getKeyIdSet(i);
                    s && s.size && s.forEach((s=>{
                        const n = "usable"
                          , r = this.keyStatusMap_.has(s) && this.keyStatusMap_.get(s) === n
                          , a = i.lastExcludeReason_ === t && i.excludeUntil === 1 / 0;
                        r ? r && a && (delete i.excludeUntil,
                        delete i.lastExcludeReason_,
                        this.logger_(`enabling playlist ${i.id} because key ID ${s} is usable`)) : (i.excludeUntil !== 1 / 0 && i.lastExcludeReason_ !== t && (i.excludeUntil = 1 / 0,
                        i.lastExcludeReason_ = t,
                        this.logger_(`excluding playlist ${i.id} because the key ID ${s} doesn't exist in the keyStatusMap or is not usable`)),
                        e++)
                    }
                    ))
                }
                )),
                e >= this.mainPlaylistLoader_.main.playlists.length && this.mainPlaylistLoader_.main.playlists.forEach((e=>{
                    const i = e && e.attributes && e.attributes.RESOLUTION && e.attributes.RESOLUTION.height < 720
                      , s = e.excludeUntil === 1 / 0 && e.lastExcludeReason_ === t;
                    i && s && (delete e.excludeUntil,
                    wn.log.warn(`enabling non-HD playlist ${e.id} because all playlists were excluded due to non-usable key IDs`))
                }
                ))
            }
            addKeyStatus_(e, t) {
                const i = ("string" === typeof e ? e : (e=>{
                    const t = new Uint8Array(e);
                    return Array.from(t).map((e=>e.toString(16).padStart(2, "0"))).join("")
                }
                )(e)).slice(0, 32).toLowerCase();
                this.logger_(`KeyStatus '${t}' with key ID ${i} added to the keyStatusMap`),
                this.keyStatusMap_.set(i, t)
            }
            updatePlaylistByKeyStatus(e, t) {
                this.addKeyStatus_(e, t),
                this.waitingForFastQualityPlaylistReceived_ || this.excludeNonUsableThenChangePlaylist_(),
                this.mainPlaylistLoader_.off("loadedplaylist", this.excludeNonUsableThenChangePlaylist_.bind(this)),
                this.mainPlaylistLoader_.on("loadedplaylist", this.excludeNonUsableThenChangePlaylist_.bind(this))
            }
            excludeNonUsableThenChangePlaylist_() {
                this.excludeNonUsablePlaylistsByKeyId_(),
                this.fastQualityChange_()
            }
        }
        class Ro {
            constructor(e, t, i) {
                const {playlistController_: s} = e
                  , n = s.fastQualityChange_.bind(s);
                if (t.attributes) {
                    const e = t.attributes.RESOLUTION;
                    this.width = e && e.width,
                    this.height = e && e.height,
                    this.bandwidth = t.attributes.BANDWIDTH,
                    this.frameRate = t.attributes["FRAME-RATE"]
                }
                var r, a, o;
                this.codecs = ka(s.main(), t),
                this.playlist = t,
                this.id = i,
                this.enabled = (r = e.playlists,
                a = t.id,
                o = n,
                e=>{
                    const t = r.main.playlists[a]
                      , i = tr(t)
                      , s = ir(t);
                    return "undefined" === typeof e ? s : (e ? delete t.disabled : t.disabled = !0,
                    e === s || i || (o(),
                    e ? r.trigger("renditionenabled") : r.trigger("renditiondisabled")),
                    e)
                }
                )
            }
        }
        const Uo = ["seeking", "seeked", "pause", "playing", "error"];
        class Bo {
            constructor(e) {
                this.playlistController_ = e.playlistController,
                this.tech_ = e.tech,
                this.seekable = e.seekable,
                this.allowSeeksWithinUnsafeLiveWindow = e.allowSeeksWithinUnsafeLiveWindow,
                this.liveRangeSafeTimeDelta = e.liveRangeSafeTimeDelta,
                this.media = e.media,
                this.consecutiveUpdates = 0,
                this.lastRecordedTime = null,
                this.checkCurrentTimeTimeout_ = null,
                this.logger_ = On("PlaybackWatcher"),
                this.logger_("initialize");
                const t = ()=>this.monitorCurrentTime_()
                  , i = ()=>this.monitorCurrentTime_()
                  , s = ()=>this.techWaiting_()
                  , r = ()=>this.resetTimeUpdate_()
                  , a = this.playlistController_
                  , o = ["main", "subtitle", "audio"]
                  , l = {};
                o.forEach((e=>{
                    l[e] = {
                        reset: ()=>this.resetSegmentDownloads_(e),
                        updateend: ()=>this.checkSegmentDownloads_(e)
                    },
                    a[`${e}SegmentLoader_`].on("appendsdone", l[e].updateend),
                    a[`${e}SegmentLoader_`].on("playlistupdate", l[e].reset),
                    this.tech_.on(["seeked", "seeking"], l[e].reset)
                }
                ));
                const h = e=>{
                    ["main", "audio"].forEach((t=>{
                        a[`${t}SegmentLoader_`][e]("appended", this.seekingAppendCheck_)
                    }
                    ))
                }
                ;
                this.seekingAppendCheck_ = ()=>{
                    this.fixesBadSeeks_() && (this.consecutiveUpdates = 0,
                    this.lastRecordedTime = this.tech_.currentTime(),
                    h("off"))
                }
                ,
                this.clearSeekingAppendCheck_ = ()=>h("off"),
                this.watchForBadSeeking_ = ()=>{
                    this.clearSeekingAppendCheck_(),
                    h("on")
                }
                ,
                this.tech_.on("seeked", this.clearSeekingAppendCheck_),
                this.tech_.on("seeking", this.watchForBadSeeking_),
                this.tech_.on("waiting", s),
                this.tech_.on(Uo, r),
                this.tech_.on("canplay", i),
                this.tech_.one("play", t),
                this.dispose = ()=>{
                    this.clearSeekingAppendCheck_(),
                    this.logger_("dispose"),
                    this.tech_.off("waiting", s),
                    this.tech_.off(Uo, r),
                    this.tech_.off("canplay", i),
                    this.tech_.off("play", t),
                    this.tech_.off("seeking", this.watchForBadSeeking_),
                    this.tech_.off("seeked", this.clearSeekingAppendCheck_),
                    o.forEach((e=>{
                        a[`${e}SegmentLoader_`].off("appendsdone", l[e].updateend),
                        a[`${e}SegmentLoader_`].off("playlistupdate", l[e].reset),
                        this.tech_.off(["seeked", "seeking"], l[e].reset)
                    }
                    )),
                    this.checkCurrentTimeTimeout_ && n().clearTimeout(this.checkCurrentTimeTimeout_),
                    this.resetTimeUpdate_()
                }
            }
            monitorCurrentTime_() {
                this.checkCurrentTime_(),
                this.checkCurrentTimeTimeout_ && n().clearTimeout(this.checkCurrentTimeTimeout_),
                this.checkCurrentTimeTimeout_ = n().setTimeout(this.monitorCurrentTime_.bind(this), 250)
            }
            resetSegmentDownloads_(e) {
                const t = this.playlistController_[`${e}SegmentLoader_`];
                this[`${e}StalledDownloads_`] > 0 && this.logger_(`resetting possible stalled download count for ${e} loader`),
                this[`${e}StalledDownloads_`] = 0,
                this[`${e}Buffered_`] = t.buffered_()
            }
            checkSegmentDownloads_(e) {
                const t = this.playlistController_
                  , i = t[`${e}SegmentLoader_`]
                  , s = i.buffered_()
                  , n = function(e, t) {
                    if (e === t)
                        return !1;
                    if (!e && t || !t && e)
                        return !0;
                    if (e.length !== t.length)
                        return !0;
                    for (let i = 0; i < e.length; i++)
                        if (e.start(i) !== t.start(i) || e.end(i) !== t.end(i))
                            return !0;
                    return !1
                }(this[`${e}Buffered_`], s);
                this[`${e}Buffered_`] = s,
                n ? this.resetSegmentDownloads_(e) : (this[`${e}StalledDownloads_`]++,
                this.logger_(`found #${this[`${e}StalledDownloads_`]} ${e} appends that did not increase buffer (possible stalled download)`, {
                    playlistId: i.playlist_ && i.playlist_.id,
                    buffered: qn(s)
                }),
                this[`${e}StalledDownloads_`] < 10 || (this.logger_(`${e} loader stalled download exclusion`),
                this.resetSegmentDownloads_(e),
                this.tech_.trigger({
                    type: "usage",
                    name: `vhs-${e}-download-exclusion`
                }),
                "subtitle" !== e && t.excludePlaylist({
                    error: {
                        message: `Excessive ${e} segment downloading detected.`
                    },
                    playlistExclusionDuration: 1 / 0
                })))
            }
            checkCurrentTime_() {
                if (this.tech_.paused() || this.tech_.seeking())
                    return;
                const e = this.tech_.currentTime()
                  , t = this.tech_.buffered();
                if (this.lastRecordedTime === e && (!t.length || e + Bn >= t.end(t.length - 1)))
                    return this.techWaiting_();
                this.consecutiveUpdates >= 5 && e === this.lastRecordedTime ? (this.consecutiveUpdates++,
                this.waiting_()) : e === this.lastRecordedTime ? this.consecutiveUpdates++ : (this.consecutiveUpdates = 0,
                this.lastRecordedTime = e)
            }
            resetTimeUpdate_() {
                this.consecutiveUpdates = 0
            }
            fixesBadSeeks_() {
                if (!this.tech_.seeking())
                    return !1;
                const e = this.seekable()
                  , t = this.tech_.currentTime();
                let i;
                if (this.afterSeekableWindow_(e, t, this.media(), this.allowSeeksWithinUnsafeLiveWindow)) {
                    i = e.end(e.length - 1)
                }
                if (this.beforeSeekableWindow_(e, t)) {
                    const t = e.start(0);
                    i = t + (t === e.end(0) ? 0 : Bn)
                }
                if ("undefined" !== typeof i)
                    return this.logger_(`Trying to seek outside of seekable at time ${t} with seekable range ${$n(e)}. Seeking to ${i}.`),
                    this.tech_.setCurrentTime(i),
                    !0;
                const s = this.playlistController_.sourceUpdater_
                  , n = this.tech_.buffered()
                  , r = s.audioBuffer ? s.audioBuffered() : null
                  , a = s.videoBuffer ? s.videoBuffered() : null
                  , o = this.media()
                  , l = o.partTargetDuration ? o.partTargetDuration : 2 * (o.targetDuration - Un)
                  , h = [r, a];
                for (let u = 0; u < h.length; u++) {
                    if (!h[u])
                        continue;
                    if (Vn(h[u], t) < l)
                        return !1
                }
                const d = jn(n, t);
                return 0 !== d.length && (i = d.start(0) + Bn,
                this.logger_(`Buffered region starts (${d.start(0)})  just beyond seek point (${t}). Seeking to ${i}.`),
                this.tech_.setCurrentTime(i),
                !0)
            }
            waiting_() {
                if (this.techWaiting_())
                    return;
                const e = this.tech_.currentTime()
                  , t = this.tech_.buffered()
                  , i = Fn(t, e);
                return i.length && e + 3 <= i.end(0) ? (this.resetTimeUpdate_(),
                this.tech_.setCurrentTime(e),
                this.logger_(`Stopped at ${e} while inside a buffered region [${i.start(0)} -> ${i.end(0)}]. Attempting to resume playback by seeking to the current time.`),
                void this.tech_.trigger({
                    type: "usage",
                    name: "vhs-unknown-waiting"
                })) : void 0
            }
            techWaiting_() {
                const e = this.seekable()
                  , t = this.tech_.currentTime();
                if (this.tech_.seeking())
                    return !0;
                if (this.beforeSeekableWindow_(e, t)) {
                    const i = e.end(e.length - 1);
                    return this.logger_(`Fell out of live window at time ${t}. Seeking to live point (seekable end) ${i}`),
                    this.resetTimeUpdate_(),
                    this.tech_.setCurrentTime(i),
                    this.tech_.trigger({
                        type: "usage",
                        name: "vhs-live-resync"
                    }),
                    !0
                }
                const i = this.tech_.vhs.playlistController_.sourceUpdater_
                  , s = this.tech_.buffered();
                if (this.videoUnderflow_({
                    audioBuffered: i.audioBuffered(),
                    videoBuffered: i.videoBuffered(),
                    currentTime: t
                }))
                    return this.resetTimeUpdate_(),
                    this.tech_.setCurrentTime(t),
                    this.tech_.trigger({
                        type: "usage",
                        name: "vhs-video-underflow"
                    }),
                    !0;
                const n = jn(s, t);
                return n.length > 0 && (this.logger_(`Stopped at ${t} and seeking to ${n.start(0)}`),
                this.resetTimeUpdate_(),
                this.skipTheGap_(t),
                !0)
            }
            afterSeekableWindow_(e, t, i, s=!1) {
                if (!e.length)
                    return !1;
                let n = e.end(e.length - 1) + Bn;
                const r = !i.endList
                  , a = "number" === typeof i.partTargetDuration;
                return r && (a || s) && (n = e.end(e.length - 1) + 3 * i.targetDuration),
                t > n
            }
            beforeSeekableWindow_(e, t) {
                return !!(e.length && e.start(0) > 0 && t < e.start(0) - this.liveRangeSafeTimeDelta)
            }
            videoUnderflow_({videoBuffered: e, audioBuffered: t, currentTime: i}) {
                if (!e)
                    return;
                let s;
                if (e.length && t.length) {
                    const n = Fn(e, i - 3)
                      , r = Fn(e, i)
                      , a = Fn(t, i);
                    a.length && !r.length && n.length && (s = {
                        start: n.end(0),
                        end: a.end(0)
                    })
                } else {
                    jn(e, i).length || (s = this.gapFromVideoUnderflow_(e, i))
                }
                return !!s && (this.logger_(`Encountered a gap in video from ${s.start} to ${s.end}. Seeking to current time ${i}`),
                !0)
            }
            skipTheGap_(e) {
                const t = this.tech_.buffered()
                  , i = this.tech_.currentTime()
                  , s = jn(t, i);
                this.resetTimeUpdate_(),
                0 !== s.length && i === e && (this.logger_("skipTheGap_:", "currentTime:", i, "scheduled currentTime:", e, "nextRange start:", s.start(0)),
                this.tech_.setCurrentTime(s.start(0) + Un),
                this.tech_.trigger({
                    type: "usage",
                    name: "vhs-gap-skip"
                }))
            }
            gapFromVideoUnderflow_(e, t) {
                const i = function(e) {
                    if (e.length < 2)
                        return Rn();
                    const t = [];
                    for (let i = 1; i < e.length; i++) {
                        const s = e.end(i - 1)
                          , n = e.start(i);
                        t.push([s, n])
                    }
                    return Rn(t)
                }(e);
                for (let s = 0; s < i.length; s++) {
                    const e = i.start(s)
                      , n = i.end(s);
                    if (t - e < 4 && t - e > 2)
                        return {
                            start: e,
                            end: n
                        }
                }
                return null
            }
        }
        const No = {
            errorInterval: 30,
            getSource(e) {
                return e(this.tech({
                    IWillNotUseThisInPlugins: !0
                }).currentSource_ || this.currentSource())
            }
        }
          , Fo = function(e, t) {
            let i = 0
              , s = 0;
            const n = Mn(No, t);
            e.ready((()=>{
                e.trigger({
                    type: "usage",
                    name: "vhs-error-reload-initialized"
                })
            }
            ));
            const r = function() {
                s && e.currentTime(s)
            }
              , a = function(t) {
                null !== t && void 0 !== t && (s = e.duration() !== 1 / 0 && e.currentTime() || 0,
                e.one("loadedmetadata", r),
                e.src(t),
                e.trigger({
                    type: "usage",
                    name: "vhs-error-reload"
                }),
                e.play())
            }
              , o = function() {
                if (Date.now() - i < 1e3 * n.errorInterval)
                    e.trigger({
                        type: "usage",
                        name: "vhs-error-reload-canceled"
                    });
                else {
                    if (n.getSource && "function" === typeof n.getSource)
                        return i = Date.now(),
                        n.getSource.call(e, a);
                    wn.log.error("ERROR: reloadSourceOnError - The option getSource must be a function!")
                }
            }
              , l = function() {
                e.off("loadedmetadata", r),
                e.off("error", o),
                e.off("dispose", l)
            };
            e.on("error", o),
            e.on("dispose", l),
            e.reloadSourceOnError = function(t) {
                l(),
                Fo(e, t)
            }
        }
          , jo = function(e) {
            Fo(this, e)
        };
        var $o = "3.10.0";
        const qo = {
            PlaylistLoader: kr,
            Playlist: lr,
            utils: Rr,
            STANDARD_PLAYLIST_SELECTOR: Aa,
            INITIAL_PLAYLIST_SELECTOR: function() {
                const e = this.playlists.main.playlists.filter(lr.isEnabled);
                xa(e, ((e,t)=>Ia(e, t)));
                return e.filter((e=>!!ka(this.playlists.main, e).video))[0] || null
            },
            lastBandwidthSelector: Aa,
            movingAverageBandwidthSelector: function(e) {
                let t = -1
                  , i = -1;
                if (e < 0 || e > 1)
                    throw new Error("Moving average bandwidth decay must be between 0 and 1.");
                return function() {
                    const s = this.useDevicePixelRatio && n().devicePixelRatio || 1;
                    return t < 0 && (t = this.systemBandwidth,
                    i = this.systemBandwidth),
                    this.systemBandwidth > 0 && this.systemBandwidth !== i && (t = e * this.systemBandwidth + (1 - e) * t,
                    i = this.systemBandwidth),
                    Pa(this.playlists.main, t, parseInt(wa(this.tech_.el(), "width"), 10) * s, parseInt(wa(this.tech_.el(), "height"), 10) * s, this.limitRenditionByPlayerDimensions, this.playlistController_)
                }
            },
            comparePlaylistBandwidth: Ia,
            comparePlaylistResolution: function(e, t) {
                let i, s;
                return e.attributes.RESOLUTION && e.attributes.RESOLUTION.width && (i = e.attributes.RESOLUTION.width),
                i = i || n().Number.MAX_VALUE,
                t.attributes.RESOLUTION && t.attributes.RESOLUTION.width && (s = t.attributes.RESOLUTION.width),
                s = s || n().Number.MAX_VALUE,
                i === s && e.attributes.BANDWIDTH && t.attributes.BANDWIDTH ? e.attributes.BANDWIDTH - t.attributes.BANDWIDTH : i - s
            },
            xhr: wr()
        };
        Object.keys(zr).forEach((e=>{
            Object.defineProperty(qo, e, {
                get: ()=>(wn.log.warn(`using Vhs.${e} is UNSAFE be sure you know what you are doing`),
                zr[e]),
                set(t) {
                    wn.log.warn(`using Vhs.${e} is UNSAFE be sure you know what you are doing`),
                    "number" !== typeof t || t < 0 ? wn.log.warn(`value of Vhs.${e} must be greater than or equal to 0`) : zr[e] = t
                }
            })
        }
        ));
        const Ho = "videojs-vhs"
          , Vo = function(e, t) {
            const i = t.media();
            let s = -1;
            for (let n = 0; n < e.length; n++)
                if (e[n].id === i.id) {
                    s = n;
                    break
                }
            e.selectedIndex_ = s,
            e.trigger({
                selectedIndex: s,
                type: "change"
            })
        };
        qo.canPlaySource = function() {
            return wn.log.warn("VHS is no longer a tech. Please remove it from your player's techOrder.")
        }
        ;
        const zo = ({player: e, sourceKeySystems: t, audioMedia: i, mainPlaylists: s})=>{
            if (!e.eme.initializeMediaKeys)
                return Promise.resolve();
            const n = ((e,t)=>e.reduce(((e,i)=>{
                if (!i.contentProtection)
                    return e;
                const s = t.reduce(((e,t)=>{
                    const s = i.contentProtection[t];
                    return s && s.pssh && (e[t] = {
                        pssh: s.pssh
                    }),
                    e
                }
                ), {});
                return Object.keys(s).length && e.push(s),
                e
            }
            ), []))(i ? s.concat([i]) : s, Object.keys(t))
              , r = []
              , a = [];
            return n.forEach((t=>{
                a.push(new Promise(((t,i)=>{
                    e.tech_.one("keysessioncreated", t)
                }
                ))),
                r.push(new Promise(((i,s)=>{
                    e.eme.initializeMediaKeys({
                        keySystems: t
                    }, (e=>{
                        e ? s(e) : i()
                    }
                    ))
                }
                )))
            }
            )),
            Promise.race([Promise.all(r), Promise.race(a)])
        }
          , Wo = ({player: e, sourceKeySystems: t, media: i, audioMedia: s})=>{
            const n = ((e,t,i)=>{
                if (!e)
                    return e;
                let s = {};
                t && t.attributes && t.attributes.CODECS && (s = ba((0,
                y.kS)(t.attributes.CODECS))),
                i && i.attributes && i.attributes.CODECS && (s.audio = i.attributes.CODECS);
                const n = (0,
                y._5)(s.video)
                  , r = (0,
                y._5)(s.audio)
                  , a = {};
                for (const o in e)
                    a[o] = {},
                    r && (a[o].audioContentType = r),
                    n && (a[o].videoContentType = n),
                    t.contentProtection && t.contentProtection[o] && t.contentProtection[o].pssh && (a[o].pssh = t.contentProtection[o].pssh),
                    "string" === typeof e[o] && (a[o].url = e[o]);
                return Mn(e, a)
            }
            )(t, i, s);
            return !!n && (e.currentSource().keySystems = n,
            !(n && !e.eme) || (wn.log.warn("DRM encrypted source cannot be decrypted without a DRM plugin"),
            !1))
        }
          , Go = ()=>{
            if (!n().localStorage)
                return null;
            const e = n().localStorage.getItem(Ho);
            if (!e)
                return null;
            try {
                return JSON.parse(e)
            } catch (t) {
                return null
            }
        }
          , Ko = (e,t)=>{
            e._requestCallbackSet || (e._requestCallbackSet = new Set),
            e._requestCallbackSet.add(t)
        }
          , Qo = (e,t)=>{
            e._responseCallbackSet || (e._responseCallbackSet = new Set),
            e._responseCallbackSet.add(t)
        }
          , Xo = (e,t)=>{
            e._requestCallbackSet && (e._requestCallbackSet.delete(t),
            e._requestCallbackSet.size || delete e._requestCallbackSet)
        }
          , Yo = (e,t)=>{
            e._responseCallbackSet && (e._responseCallbackSet.delete(t),
            e._responseCallbackSet.size || delete e._responseCallbackSet)
        }
        ;
        qo.supportsNativeHls = function() {
            if (!a() || !a().createElement)
                return !1;
            const e = a().createElement("video");
            if (!wn.getTech("Html5").isSupported())
                return !1;
            return ["application/vnd.apple.mpegurl", "audio/mpegurl", "audio/x-mpegurl", "application/x-mpegurl", "video/x-mpegurl", "video/mpegurl", "application/mpegurl"].some((function(t) {
                return /maybe|probably/i.test(e.canPlayType(t))
            }
            ))
        }(),
        qo.supportsNativeDash = !!(a() && a().createElement && wn.getTech("Html5").isSupported()) && /maybe|probably/i.test(a().createElement("video").canPlayType("application/dash+xml")),
        qo.supportsTypeNatively = e=>"hls" === e ? qo.supportsNativeHls : "dash" === e && qo.supportsNativeDash,
        qo.isSupported = function() {
            return wn.log.warn("VHS is no longer a tech. Please remove it from your player's techOrder.")
        }
        ,
        qo.xhr.onRequest = function(e) {
            Ko(qo.xhr, e)
        }
        ,
        qo.xhr.onResponse = function(e) {
            Qo(qo.xhr, e)
        }
        ,
        qo.xhr.offRequest = function(e) {
            Xo(qo.xhr, e)
        }
        ,
        qo.xhr.offResponse = function(e) {
            Yo(qo.xhr, e)
        }
        ;
        const Jo = wn.getComponent("Component");
        class Zo extends Jo {
            constructor(e, t, i) {
                if (super(t, i.vhs),
                "number" === typeof i.initialBandwidth && (this.options_.bandwidth = i.initialBandwidth),
                this.logger_ = On("VhsHandler"),
                t.options_ && t.options_.playerId) {
                    const e = wn.getPlayer(t.options_.playerId);
                    this.player_ = e
                }
                if (this.tech_ = t,
                this.source_ = e,
                this.stats = {},
                this.ignoreNextSeekingEvent_ = !1,
                this.setOptions_(),
                this.options_.overrideNative && t.overrideNativeAudioTracks && t.overrideNativeVideoTracks)
                    t.overrideNativeAudioTracks(!0),
                    t.overrideNativeVideoTracks(!0);
                else if (this.options_.overrideNative && (t.featuresNativeVideoTracks || t.featuresNativeAudioTracks))
                    throw new Error("Overriding native VHS requires emulated tracks. See https://git.io/vMpjB");
                this.on(a(), ["fullscreenchange", "webkitfullscreenchange", "mozfullscreenchange", "MSFullscreenChange"], (e=>{
                    const t = a().fullscreenElement || a().webkitFullscreenElement || a().mozFullScreenElement || a().msFullscreenElement;
                    t && t.contains(this.tech_.el()) ? this.playlistController_.fastQualityChange_() : this.playlistController_.checkABR_()
                }
                )),
                this.on(this.tech_, "seeking", (function() {
                    this.ignoreNextSeekingEvent_ ? this.ignoreNextSeekingEvent_ = !1 : this.setCurrentTime(this.tech_.currentTime())
                }
                )),
                this.on(this.tech_, "error", (function() {
                    this.tech_.error() && this.playlistController_ && this.playlistController_.pauseLoading()
                }
                )),
                this.on(this.tech_, "play", this.play)
            }
            setOptions_(e={}) {
                if (this.options_ = Mn(this.options_, e),
                this.options_.withCredentials = this.options_.withCredentials || !1,
                this.options_.limitRenditionByPlayerDimensions = !1 !== this.options_.limitRenditionByPlayerDimensions,
                this.options_.useDevicePixelRatio = this.options_.useDevicePixelRatio || !1,
                this.options_.useBandwidthFromLocalStorage = "undefined" !== typeof this.source_.useBandwidthFromLocalStorage ? this.source_.useBandwidthFromLocalStorage : this.options_.useBandwidthFromLocalStorage || !1,
                this.options_.useForcedSubtitles = this.options_.useForcedSubtitles || !1,
                this.options_.useNetworkInformationApi = this.options_.useNetworkInformationApi || !1,
                this.options_.useDtsForTimestampOffset = this.options_.useDtsForTimestampOffset || !1,
                this.options_.customTagParsers = this.options_.customTagParsers || [],
                this.options_.customTagMappers = this.options_.customTagMappers || [],
                this.options_.cacheEncryptionKeys = this.options_.cacheEncryptionKeys || !1,
                this.options_.llhls = !1 !== this.options_.llhls,
                this.options_.bufferBasedABR = this.options_.bufferBasedABR || !1,
                "number" !== typeof this.options_.playlistExclusionDuration && (this.options_.playlistExclusionDuration = 60),
                "number" !== typeof this.options_.bandwidth && this.options_.useBandwidthFromLocalStorage) {
                    const e = Go();
                    e && e.bandwidth && (this.options_.bandwidth = e.bandwidth,
                    this.tech_.trigger({
                        type: "usage",
                        name: "vhs-bandwidth-from-local-storage"
                    })),
                    e && e.throughput && (this.options_.throughput = e.throughput,
                    this.tech_.trigger({
                        type: "usage",
                        name: "vhs-throughput-from-local-storage"
                    }))
                }
                "number" !== typeof this.options_.bandwidth && (this.options_.bandwidth = zr.INITIAL_BANDWIDTH),
                this.options_.enableLowInitialPlaylist = this.options_.enableLowInitialPlaylist && this.options_.bandwidth === zr.INITIAL_BANDWIDTH,
                ["withCredentials", "useDevicePixelRatio", "limitRenditionByPlayerDimensions", "bandwidth", "customTagParsers", "customTagMappers", "cacheEncryptionKeys", "playlistSelector", "initialPlaylistSelector", "bufferBasedABR", "liveRangeSafeTimeDelta", "llhls", "useForcedSubtitles", "useNetworkInformationApi", "useDtsForTimestampOffset", "exactManifestTimings", "leastPixelDiffSelector"].forEach((e=>{
                    "undefined" !== typeof this.source_[e] && (this.options_[e] = this.source_[e])
                }
                )),
                this.limitRenditionByPlayerDimensions = this.options_.limitRenditionByPlayerDimensions,
                this.useDevicePixelRatio = this.options_.useDevicePixelRatio
            }
            setOptions(e={}) {
                this.setOptions_(e)
            }
            src(e, t) {
                if (!e)
                    return;
                var i;
                this.setOptions_(),
                this.options_.src = 0 === (i = this.source_.src).toLowerCase().indexOf("data:application/vnd.videojs.vhs+json,") ? JSON.parse(i.substring(i.indexOf(",") + 1)) : i,
                this.options_.tech = this.tech_,
                this.options_.externVhs = qo,
                this.options_.sourceType = (0,
                v.t)(t),
                this.options_.seekTo = e=>{
                    this.tech_.setCurrentTime(e)
                }
                ,
                this.playlistController_ = new Mo(this.options_);
                const s = Mn({
                    liveRangeSafeTimeDelta: Bn
                }, this.options_, {
                    seekable: ()=>this.seekable(),
                    media: ()=>this.playlistController_.media(),
                    playlistController: this.playlistController_
                });
                this.playbackWatcher_ = new Bo(s),
                this.playlistController_.on("error", (()=>{
                    const e = wn.players[this.tech_.options_.playerId];
                    let t = this.playlistController_.error;
                    "object" !== typeof t || t.code ? "string" === typeof t && (t = {
                        message: t,
                        code: 3
                    }) : t.code = 3,
                    e.error(t)
                }
                ));
                const r = this.options_.bufferBasedABR ? qo.movingAverageBandwidthSelector(.55) : qo.STANDARD_PLAYLIST_SELECTOR;
                this.playlistController_.selectPlaylist = this.selectPlaylist ? this.selectPlaylist.bind(this) : r.bind(this),
                this.playlistController_.selectInitialPlaylist = qo.INITIAL_PLAYLIST_SELECTOR.bind(this),
                this.playlists = this.playlistController_.mainPlaylistLoader_,
                this.mediaSource = this.playlistController_.mediaSource,
                Object.defineProperties(this, {
                    selectPlaylist: {
                        get() {
                            return this.playlistController_.selectPlaylist
                        },
                        set(e) {
                            this.playlistController_.selectPlaylist = e.bind(this)
                        }
                    },
                    throughput: {
                        get() {
                            return this.playlistController_.mainSegmentLoader_.throughput.rate
                        },
                        set(e) {
                            this.playlistController_.mainSegmentLoader_.throughput.rate = e,
                            this.playlistController_.mainSegmentLoader_.throughput.count = 1
                        }
                    },
                    bandwidth: {
                        get() {
                            let e = this.playlistController_.mainSegmentLoader_.bandwidth;
                            const t = n().navigator.connection || n().navigator.mozConnection || n().navigator.webkitConnection
                              , i = 1e7;
                            if (this.options_.useNetworkInformationApi && t) {
                                const s = 1e3 * t.downlink * 1e3;
                                e = s >= i && e >= i ? Math.max(e, s) : s
                            }
                            return e
                        },
                        set(e) {
                            this.playlistController_.mainSegmentLoader_.bandwidth = e,
                            this.playlistController_.mainSegmentLoader_.throughput = {
                                rate: 0,
                                count: 0
                            }
                        }
                    },
                    systemBandwidth: {
                        get() {
                            const e = 1 / (this.bandwidth || 1);
                            let t;
                            t = this.throughput > 0 ? 1 / this.throughput : 0;
                            return Math.floor(1 / (e + t))
                        },
                        set() {
                            wn.log.error('The "systemBandwidth" property is read-only')
                        }
                    }
                }),
                this.options_.bandwidth && (this.bandwidth = this.options_.bandwidth),
                this.options_.throughput && (this.throughput = this.options_.throughput),
                Object.defineProperties(this.stats, {
                    bandwidth: {
                        get: ()=>this.bandwidth || 0,
                        enumerable: !0
                    },
                    mediaRequests: {
                        get: ()=>this.playlistController_.mediaRequests_() || 0,
                        enumerable: !0
                    },
                    mediaRequestsAborted: {
                        get: ()=>this.playlistController_.mediaRequestsAborted_() || 0,
                        enumerable: !0
                    },
                    mediaRequestsTimedout: {
                        get: ()=>this.playlistController_.mediaRequestsTimedout_() || 0,
                        enumerable: !0
                    },
                    mediaRequestsErrored: {
                        get: ()=>this.playlistController_.mediaRequestsErrored_() || 0,
                        enumerable: !0
                    },
                    mediaTransferDuration: {
                        get: ()=>this.playlistController_.mediaTransferDuration_() || 0,
                        enumerable: !0
                    },
                    mediaBytesTransferred: {
                        get: ()=>this.playlistController_.mediaBytesTransferred_() || 0,
                        enumerable: !0
                    },
                    mediaSecondsLoaded: {
                        get: ()=>this.playlistController_.mediaSecondsLoaded_() || 0,
                        enumerable: !0
                    },
                    mediaAppends: {
                        get: ()=>this.playlistController_.mediaAppends_() || 0,
                        enumerable: !0
                    },
                    mainAppendsToLoadedData: {
                        get: ()=>this.playlistController_.mainAppendsToLoadedData_() || 0,
                        enumerable: !0
                    },
                    audioAppendsToLoadedData: {
                        get: ()=>this.playlistController_.audioAppendsToLoadedData_() || 0,
                        enumerable: !0
                    },
                    appendsToLoadedData: {
                        get: ()=>this.playlistController_.appendsToLoadedData_() || 0,
                        enumerable: !0
                    },
                    timeToLoadedData: {
                        get: ()=>this.playlistController_.timeToLoadedData_() || 0,
                        enumerable: !0
                    },
                    buffered: {
                        get: ()=>qn(this.tech_.buffered()),
                        enumerable: !0
                    },
                    currentTime: {
                        get: ()=>this.tech_.currentTime(),
                        enumerable: !0
                    },
                    currentSource: {
                        get: ()=>this.tech_.currentSource_,
                        enumerable: !0
                    },
                    currentTech: {
                        get: ()=>this.tech_.name_,
                        enumerable: !0
                    },
                    duration: {
                        get: ()=>this.tech_.duration(),
                        enumerable: !0
                    },
                    main: {
                        get: ()=>this.playlists.main,
                        enumerable: !0
                    },
                    playerDimensions: {
                        get: ()=>this.tech_.currentDimensions(),
                        enumerable: !0
                    },
                    seekable: {
                        get: ()=>qn(this.tech_.seekable()),
                        enumerable: !0
                    },
                    timestamp: {
                        get: ()=>Date.now(),
                        enumerable: !0
                    },
                    videoPlaybackQuality: {
                        get: ()=>this.tech_.getVideoPlaybackQuality(),
                        enumerable: !0
                    }
                }),
                this.tech_.one("canplay", this.playlistController_.setupFirstPlay.bind(this.playlistController_)),
                this.tech_.on("bandwidthupdate", (()=>{
                    this.options_.useBandwidthFromLocalStorage && (e=>{
                        if (!n().localStorage)
                            return !1;
                        let t = Go();
                        t = t ? Mn(t, e) : e;
                        try {
                            n().localStorage.setItem(Ho, JSON.stringify(t))
                        } catch (i) {
                            return !1
                        }
                    }
                    )({
                        bandwidth: this.bandwidth,
                        throughput: Math.round(this.throughput)
                    })
                }
                )),
                this.playlistController_.on("selectedinitialmedia", (()=>{
                    var e;
                    (e = this).representations = ()=>{
                        const t = e.playlistController_.main()
                          , i = or(t) ? e.playlistController_.getAudioTrackPlaylists_() : t.playlists;
                        return i ? i.filter((e=>!tr(e))).map(((t,i)=>new Ro(e,t,t.id))) : []
                    }
                }
                )),
                this.playlistController_.sourceUpdater_.on("createdsourcebuffers", (()=>{
                    this.setupEme_()
                }
                )),
                this.on(this.playlistController_, "progress", (function() {
                    this.tech_.trigger("progress")
                }
                )),
                this.on(this.playlistController_, "firstplay", (function() {
                    this.ignoreNextSeekingEvent_ = !0
                }
                )),
                this.setupQualityLevels_(),
                this.tech_.el() && (this.mediaSourceUrl_ = n().URL.createObjectURL(this.playlistController_.mediaSource),
                this.tech_.src(this.mediaSourceUrl_))
            }
            createKeySessions_() {
                const e = this.playlistController_.mediaTypes_.AUDIO.activePlaylistLoader;
                this.logger_("waiting for EME key session creation"),
                zo({
                    player: this.player_,
                    sourceKeySystems: this.source_.keySystems,
                    audioMedia: e && e.media(),
                    mainPlaylists: this.playlists.main.playlists
                }).then((()=>{
                    this.logger_("created EME key session"),
                    this.playlistController_.sourceUpdater_.initializedEme()
                }
                )).catch((e=>{
                    this.logger_("error while creating EME key session", e),
                    this.player_.error({
                        message: "Failed to initialize media keys for EME",
                        code: 3
                    })
                }
                ))
            }
            handleWaitingForKey_() {
                this.logger_("waitingforkey fired, attempting to create any new key sessions"),
                this.createKeySessions_()
            }
            setupEme_() {
                const e = this.playlistController_.mediaTypes_.AUDIO.activePlaylistLoader
                  , t = Wo({
                    player: this.player_,
                    sourceKeySystems: this.source_.keySystems,
                    media: this.playlists.media(),
                    audioMedia: e && e.media()
                });
                this.player_.tech_.on("keystatuschange", (e=>{
                    this.playlistController_.updatePlaylistByKeyStatus(e.keyId, e.status)
                }
                )),
                this.handleWaitingForKey_ = this.handleWaitingForKey_.bind(this),
                this.player_.tech_.on("waitingforkey", this.handleWaitingForKey_),
                t ? this.createKeySessions_() : this.playlistController_.sourceUpdater_.initializedEme()
            }
            setupQualityLevels_() {
                const e = wn.players[this.tech_.options_.playerId];
                e && e.qualityLevels && !this.qualityLevels_ && (this.qualityLevels_ = e.qualityLevels(),
                this.playlistController_.on("selectedinitialmedia", (()=>{
                    !function(e, t) {
                        t.representations().forEach((t=>{
                            e.addQualityLevel(t)
                        }
                        )),
                        Vo(e, t.playlists)
                    }(this.qualityLevels_, this)
                }
                )),
                this.playlists.on("mediachange", (()=>{
                    Vo(this.qualityLevels_, this.playlists)
                }
                )))
            }
            static version() {
                return {
                    "@videojs/http-streaming": $o,
                    "mux.js": "7.0.2",
                    "mpd-parser": "1.3.0",
                    "m3u8-parser": "7.1.0",
                    "aes-decrypter": "4.0.1"
                }
            }
            version() {
                return this.constructor.version()
            }
            canChangeType() {
                return ho.canChangeType()
            }
            play() {
                this.playlistController_.play()
            }
            setCurrentTime(e) {
                this.playlistController_.setCurrentTime(e)
            }
            duration() {
                return this.playlistController_.duration()
            }
            seekable() {
                return this.playlistController_.seekable()
            }
            dispose() {
                this.playbackWatcher_ && this.playbackWatcher_.dispose(),
                this.playlistController_ && this.playlistController_.dispose(),
                this.qualityLevels_ && this.qualityLevels_.dispose(),
                this.tech_ && this.tech_.vhs && delete this.tech_.vhs,
                this.mediaSourceUrl_ && n().URL.revokeObjectURL && (n().URL.revokeObjectURL(this.mediaSourceUrl_),
                this.mediaSourceUrl_ = null),
                this.tech_ && this.tech_.off("waitingforkey", this.handleWaitingForKey_),
                super.dispose()
            }
            convertToProgramTime(e, t) {
                return Ur({
                    playlist: this.playlistController_.media(),
                    time: e,
                    callback: t
                })
            }
            seekToProgramTime(e, t, i=!0, s=2) {
                return Br({
                    programTime: e,
                    playlist: this.playlistController_.media(),
                    retryCount: s,
                    pauseAfterSeek: i,
                    seekTo: this.options_.seekTo,
                    tech: this.options_.tech,
                    callback: t
                })
            }
            setupXhrHooks_() {
                this.xhr.onRequest = e=>{
                    Ko(this.xhr, e)
                }
                ,
                this.xhr.onResponse = e=>{
                    Qo(this.xhr, e)
                }
                ,
                this.xhr.offRequest = e=>{
                    Xo(this.xhr, e)
                }
                ,
                this.xhr.offResponse = e=>{
                    Yo(this.xhr, e)
                }
                ,
                this.player_.trigger("xhr-hooks-ready")
            }
        }
        const el = {
            name: "videojs-http-streaming",
            VERSION: $o,
            canHandleSource(e, t={}) {
                const i = Mn(wn.options, t);
                return el.canPlayType(e.type, i)
            },
            handleSource(e, t, i={}) {
                const s = Mn(wn.options, i);
                return t.vhs = new Zo(e,t,s),
                t.vhs.xhr = wr(),
                t.vhs.setupXhrHooks_(),
                t.vhs.src(e.src, e.type),
                t.vhs
            },
            canPlayType(e, t) {
                const i = (0,
                v.t)(e);
                if (!i)
                    return "";
                const s = el.getOverrideNative(t);
                return !qo.supportsTypeNatively(i) || s ? "maybe" : ""
            },
            getOverrideNative(e={}) {
                const {vhs: t={}} = e
                  , i = !(wn.browser.IS_ANY_SAFARI || wn.browser.IS_IOS)
                  , {overrideNative: s=i} = t;
                return s
            }
        };
        (0,
        y.p7)("avc1.4d400d,mp4a.40.2") && wn.getTech("Html5").registerSourceHandler(el, 0),
        wn.VhsHandler = Zo,
        wn.VhsSourceHandler = el,
        wn.Vhs = qo,
        wn.use || wn.registerComponent("Vhs", qo),
        wn.options.vhs = wn.options.vhs || {},
        wn.getPlugin && wn.getPlugin("reloadSourceOnError") || wn.registerPlugin("reloadSourceOnError", jo)
    }
}]);
